"""Support-method mixin for :class:`weiss_rl.learners.impala_learner.ImpalaLearner`."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.batch_fields import (
    float_target,
    optional_batch_seat_field,
    optional_time_major_bool_field,
    optional_time_major_float_field,
    optional_time_major_index_field,
    optional_time_major_loss_mask,
    optional_time_major_seat_field,
    prepare_acting_seat_batch,
    prepare_legacy_hidden_state,
    prepare_seat_hidden_state,
    tensor_on_device,
)
from weiss_rl.learners.bootstrap import (
    current_model_bootstrap_value,
    has_raw_vtrace_inputs,
    resolve_vtrace_bootstrap_value,
)
from weiss_rl.learners.faults import (
    batch_fault_snapshot,
    collect_nonfinite_gradients,
    ensure_finite_gradients,
    ensure_finite_tensor,
    fault_dir_path,
    learner_batch_size,
    raise_for_nonfinite_gradients,
    write_numeric_fault_bundle,
)
from weiss_rl.learners.forward_time_major import ForwardTimeMajorResult
from weiss_rl.learners.forward_time_major import forward_time_major as learner_forward_time_major
from weiss_rl.learners.legal_fields import (
    has_legal_actions,
    require_actions,
    require_legal_mask,
    require_obs,
    resolve_legal_mask,
    resolve_packed_legal_actions_with_meta,
)
from weiss_rl.learners.logging import (
    build_training_metrics,
    custom_log_metrics,
    write_checkpoint_metadata,
)
from weiss_rl.learners.packed_rows import (
    packed_candidate_positions_for_rows,
    packed_legal_action_view,
    scatter_packed_candidate_values,
    slice_packed_legal_rows_with_meta,
    subset_observation_context_rows,
)
from weiss_rl.learners.structured_auxiliary import (
    active_public_heuristic_profiles,
    score_public_heuristic_target_logits,
)
from weiss_rl.learners.vtrace import VtraceMetrics, compute_vtrace_metrics


def _batch_value(batch: Any, key: str) -> Any:
    # Resolve through impala_learner so the historical helper remains the compatibility hook.
    from weiss_rl.learners import impala_learner as learner_module

    return learner_module._batch_value(batch, key)


class ImpalaSupportMixin:
    def _optimizer_for_step(self: Any) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = cast(Any, torch.optim.Adam(self._optimizer_parameter_groups(), lr=self.learning_rate))
        return cast(Optimizer, self.optimizer)

    def _optimizer_parameter_groups(self: Any) -> Any:
        model = self.model
        if model is None:
            raise ValueError("ImpalaLearner requires a model before creating an optimizer")
        adapter_names = {
            name
            for name, _parameter in model.named_parameters()
            if name
            in {
                "opponent_context_action_bias_adapter",
                "opponent_context_hidden_adapter",
                "opponent_context_recurrent_adapter",
            }
            or name.startswith("opponent_context_candidate_residual_")
        }
        multiplier = float(getattr(model, "opponent_context_adapter_lr_multiplier", 1.0))
        if bool(getattr(model, "opponent_context_adapter_train_only", False)):
            if not adapter_names:
                raise ValueError(
                    "opponent_context_adapter_train_only requires at least one trainable opponent-context adapter"
                )
            train_only_adapter_params: list[Tensor] = []
            for name, parameter in model.named_parameters():
                trainable = name in adapter_names
                parameter.requires_grad_(trainable)
                if trainable:
                    train_only_adapter_params.append(parameter)
            if not train_only_adapter_params:
                raise ValueError(
                    "opponent_context_adapter_train_only found no trainable opponent-context adapter parameters"
                )
            return [{"params": train_only_adapter_params, "lr": float(self.learning_rate) * multiplier}]
        if not adapter_names or multiplier == 1.0:
            return model.parameters()
        adapter_params: list[Tensor] = []
        base_params: list[Tensor] = []
        for name, parameter in model.named_parameters():
            if name in adapter_names:
                adapter_params.append(parameter)
            else:
                base_params.append(parameter)
        if not adapter_params:
            return model.parameters()
        return [
            {"params": base_params},
            {"params": adapter_params, "lr": float(self.learning_rate) * multiplier},
        ]

    def _forward_time_major(
        self: Any,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: LegalActionBatch | None = None,
        policy_train_mask: Tensor | None = None,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Any = None,
    ) -> ForwardTimeMajorResult:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        return learner_forward_time_major(
            model=self.model,
            compiled_model=self.compiled_model,
            obs=obs,
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            actor=actor,
            legal_actions=legal_actions,
            policy_train_mask=policy_train_mask,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
            prepare_acting_seat_batch=self._prepare_acting_seat_batch,
            prepare_legacy_hidden_state=self._prepare_legacy_hidden_state,
            prepare_seat_hidden_state=self._prepare_seat_hidden_state,
            slice_packed_legal_rows_with_meta=self._slice_packed_legal_rows_with_meta,
            packed_legal_action_view=self._packed_legal_action_view,
            subset_observation_context_rows=self._subset_observation_context_rows,
            scatter_packed_candidate_values=self._scatter_packed_candidate_values,
            record_timing_ms=self._record_timing_ms,
            active_timing_metrics=self._active_timing_metrics,
        )

    def _require_obs(self: Any, value: Any) -> Tensor:
        return require_obs(value, reference=self._model_parameter())

    def _require_actions(self: Any, value: Any, *, expected_shape: torch.Size) -> Tensor:
        return require_actions(value, expected_shape=expected_shape, reference=self._model_parameter())

    def _require_legal_mask(self: Any, value: Any, *, expected_shape: torch.Size) -> Tensor:
        return require_legal_mask(value, expected_shape=expected_shape, reference=self._model_parameter())

    def _has_legal_actions(self: Any, batch: Any) -> bool:
        return has_legal_actions(batch, batch_value=_batch_value)

    def _resolve_legal_mask(self: Any, batch: Any, *, expected_shape: torch.Size, action_dim: int) -> Tensor:
        return resolve_legal_mask(
            batch,
            expected_shape=expected_shape,
            action_dim=action_dim,
            reference=self._model_parameter(),
            batch_value=_batch_value,
        )

    def _resolve_packed_legal_actions(
        self: Any,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor] | None:
        resolved = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if resolved is None:
            return None
        return resolved[0], resolved[1]

    def _resolve_packed_legal_actions_with_meta(
        self: Any,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        return resolve_packed_legal_actions_with_meta(
            batch,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
            batch_value=_batch_value,
            supports_legal_candidate_scoring=bool(getattr(self.model, "supports_legal_candidate_scoring", False)),
        )

    def _packed_legal_action_view(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
    ) -> Any:
        return packed_legal_action_view(packed_legal)

    def _slice_packed_legal_rows_with_meta(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        return slice_packed_legal_rows_with_meta(packed_legal, row_indices)

    def _packed_candidate_positions_for_rows(
        self: Any,
        offsets: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        return packed_candidate_positions_for_rows(offsets, row_indices)

    def _scatter_packed_candidate_values(
        self: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
        subset_values: Tensor,
        *,
        fill_value: float = 0.0,
    ) -> Tensor:
        return scatter_packed_candidate_values(packed_legal, row_indices, subset_values, fill_value=fill_value)

    def _subset_observation_context_rows(
        self: Any,
        observation_context: Mapping[str, Tensor],
        row_indices: Tensor,
        *,
        row_count: int,
    ) -> dict[str, Tensor]:
        return subset_observation_context_rows(observation_context, row_indices, row_count=row_count)

    def _score_public_heuristic_target_logits(
        self: Any,
        *,
        forward_model: Any,
        obs_rows: Tensor,
        legal_actions: Any,
        observation_context: Mapping[str, Tensor] | None,
        device: torch.device,
    ) -> Tensor:
        return score_public_heuristic_target_logits(
            forward_model=forward_model,
            obs_rows=obs_rows,
            legal_actions=legal_actions,
            observation_context=observation_context,
            profiles=self.teacher_public_heuristic_profiles,
            profile_mode=self.teacher_public_heuristic_profile_mode,
            update_count=int(self.update_count),
            end_updates=int(self.teacher_public_heuristic_profiles_end_updates),
            temperature=float(self.teacher_public_heuristic_temperature),
            device=device,
        )

    def _active_teacher_public_heuristic_profiles(self: Any) -> tuple[str, ...]:
        return active_public_heuristic_profiles(
            self.teacher_public_heuristic_profiles,
            update_count=int(self.update_count),
            end_updates=int(self.teacher_public_heuristic_profiles_end_updates),
        )

    def _packed_public_heuristic_target_logits(
        self: Any,
        *,
        forward_model: Any,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        observation_context: Mapping[str, Tensor] | None,
    ) -> Tensor | None:
        total_rows = int(obs.shape[0] * obs.shape[1])
        active_rows = torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            return None
        flat_obs = obs.reshape(total_rows, obs.shape[-1])
        if int(active_rows.shape[0]) == total_rows:
            legal_actions = self._packed_legal_action_view(packed_legal)
            return self._score_public_heuristic_target_logits(
                forward_model=forward_model,
                obs_rows=flat_obs,
                legal_actions=legal_actions,
                observation_context=observation_context,
                device=flat_obs.device,
            )
        subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
        subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
        subset_obs = flat_obs.index_select(0, active_rows)
        subset_context = (
            None
            if observation_context is None
            else self._subset_observation_context_rows(
                observation_context,
                active_rows,
                row_count=total_rows,
            )
        )
        subset_target_logits = self._score_public_heuristic_target_logits(
            forward_model=forward_model,
            obs_rows=subset_obs,
            legal_actions=subset_legal_actions,
            observation_context=subset_context,
            device=subset_obs.device,
        )
        return self._scatter_packed_candidate_values(
            packed_legal,
            active_rows,
            subset_target_logits,
            fill_value=0.0,
        )

    def _has_raw_vtrace_inputs(self: Any, batch: Any) -> bool:
        return has_raw_vtrace_inputs(batch, batch_value=_batch_value)

    def _resolve_vtrace_bootstrap_value(
        self: Any,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor:
        return resolve_vtrace_bootstrap_value(
            batch,
            batch_size=batch_size,
            like=like,
            model=self.model,
            compiled_model=self.compiled_model,
            reference_parameter=self._model_parameter,
            batch_value=_batch_value,
        )

    def _current_model_bootstrap_value(
        self: Any,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor | None:
        return current_model_bootstrap_value(
            batch,
            batch_size=batch_size,
            like=like,
            model=self.model,
            compiled_model=self.compiled_model,
            reference_parameter=self._model_parameter,
            batch_value=_batch_value,
        )

    def _float_target(self: Any, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        return float_target(value, expected_shape=expected_shape, like=like, reference=self._model_parameter())

    def _optional_batch_seat_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_batch_size: int,
    ) -> Tensor | None:
        return optional_batch_seat_field(
            value,
            field_name=field_name,
            expected_batch_size=expected_batch_size,
            reference=self._model_parameter(),
        )

    def _prepare_legacy_hidden_state(self: Any, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        return prepare_legacy_hidden_state(value, batch_size=batch_size, like=like, reference=self._model_parameter())

    def _prepare_seat_hidden_state(self: Any, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        return prepare_seat_hidden_state(value, batch_size=batch_size, like=like, reference=self._model_parameter())

    def _prepare_acting_seat_batch(
        self: Any,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_seat_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_seat_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_loss_mask(
        self: Any,
        value: Any,
        *,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        return optional_time_major_loss_mask(
            value,
            expected_shape=expected_shape,
            like=like,
            reference=self._model_parameter(),
        )

    def _optional_time_major_index_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_index_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _optional_time_major_float_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        return optional_time_major_float_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            like=like,
            reference=self._model_parameter(),
        )

    def _optional_time_major_bool_field(
        self: Any,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        return optional_time_major_bool_field(
            value,
            field_name=field_name,
            expected_shape=expected_shape,
            reference=self._model_parameter(),
        )

    def _float_input(self: Any, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self: Any, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self: Any, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self: Any, value: Any, *, dtype: torch.dtype) -> Tensor:
        return tensor_on_device(value, reference=self._model_parameter(), dtype=dtype)

    def _model_parameter(self: Any) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter

    def _batch_size(self: Any, batch: Any) -> int:
        return learner_batch_size(batch, batch_value=_batch_value)

    def _fault_dir_path(self: Any) -> Path:
        return fault_dir_path(fault_dir=self.fault_dir, checkpoint_dir=self.checkpoint_dir, logs_dir=self.logs_dir)

    def _batch_fault_snapshot(self: Any, batch: Any) -> dict[str, Any]:
        return batch_fault_snapshot(batch, batch_value=_batch_value)

    def _write_numeric_fault_bundle(self: Any, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return write_numeric_fault_bundle(
            fault_dir=self._fault_dir_path(),
            stage=stage,
            update_count=self.update_count,
            policy_version=self.policy_version,
            batch_size=self._batch_size(batch),
            pass_action_id=self.pass_action_id,
            batch_snapshot=self._batch_fault_snapshot(batch),
            context=context,
        )

    def _ensure_finite_tensor(
        self: Any,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        ensure_finite_tensor(
            name, tensor, batch=batch, context=context, write_bundle=self._write_fault_bundle_for_stage
        )

    def _collect_nonfinite_gradients(self: Any, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        return collect_nonfinite_gradients(self.model, grad_norm)

    def _ensure_finite_gradients(self: Any, *, batch: Any, context: dict[str, Any], grad_norm: Tensor) -> None:
        bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
        ensure_finite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
            write_bundle=self._write_fault_bundle_for_stage,
        )

    def _raise_for_nonfinite_gradients(
        self: Any,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:
        raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
            write_bundle=self._write_fault_bundle_for_stage,
        )

    def _write_fault_bundle_for_stage(self: Any, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return self._write_numeric_fault_bundle(stage=stage, batch=batch, context=context)

    def _write_checkpoint_metadata(self: Any) -> None:
        checkpoint_metadata_path = write_checkpoint_metadata(
            checkpoint_dir=self.checkpoint_dir,
            update_count=self.update_count,
            policy_version=self.policy_version,
        )
        if checkpoint_metadata_path is not None:
            print(f"Saved checkpoint metadata: {checkpoint_metadata_path}")

    def _log_metrics(
        self: Any, update_metrics: dict[str, float], batch: Any, *, context: dict[str, Any] | None = None
    ) -> None:
        if not self.logger:
            return

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        vtrace_metrics = compute_vtrace_metrics(
            batch,
            rho_bar=self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
            c_bar=self.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = build_training_metrics(
            update_metrics=update_metrics,
            vtrace_metrics=vtrace_metrics,
            update_count=self.update_count,
            policy_version=self.policy_version,
            elapsed_seconds=elapsed,
        )
        self.logger.log(metrics)

    def _custom_log_metrics(
        self: Any,
        update_metrics: dict[str, float],
        vtrace_metrics: VtraceMetrics,
    ) -> dict[str, float]:
        return custom_log_metrics(update_metrics, vtrace_metrics)

    def get_policy_version(self: Any) -> int:
        return self.policy_version
