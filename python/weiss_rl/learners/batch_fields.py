"""Batch field and numeric-safety helpers for the IMPALA learner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.impala_helpers import _batch_value
from weiss_rl.learners.numeric_faults import LearnerNumericFaultReporter, collect_nonfinite_gradients
from weiss_rl.learners.packed_legal import (
    packed_candidate_positions_for_rows as _packed_candidate_positions_for_rows_impl,
)
from weiss_rl.learners.packed_legal import scatter_packed_candidate_values as _scatter_packed_candidate_values_impl
from weiss_rl.learners.packed_legal import slice_packed_legal_rows_with_meta as _slice_packed_legal_rows_with_meta_impl
from weiss_rl.learners.packed_legal import subset_observation_context_rows as _subset_observation_context_rows_impl
from weiss_rl.legal_actions import LegalActionBatch


class ImpalaBatchFieldsMixin:
    def _require_obs(self, value: Any) -> Tensor:
        tensor = self._float_input(value)
        if tensor.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(tensor.shape)}")
        return tensor

    def _require_actions(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._long_input(value)
        if tensor.shape != expected_shape:
            raise ValueError(f"actions must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _require_legal_mask(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._bool_input(value)
        if tensor.ndim != 3 or tensor.shape[:2] != expected_shape:
            expected = (int(expected_shape[0]), int(expected_shape[1]), "action")
            raise ValueError(f"legal_mask must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _has_legal_actions(self, batch: Any) -> bool:
        if _batch_value(batch, "legal_actions") is not None:
            return True
        if _batch_value(batch, "legal_mask") is not None:
            return True
        return _batch_value(batch, "legal_ids") is not None and _batch_value(batch, "legal_offsets") is not None

    def _resolve_legal_mask(self, batch: Any, *, expected_shape: torch.Size, action_dim: int) -> Tensor:
        legal_actions = _batch_value(batch, "legal_actions")
        if isinstance(legal_actions, LegalActionBatch):
            mask = legal_actions.to_mask(
                expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
                action_space=action_dim,
            )
            return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

        legal_mask = _batch_value(batch, "legal_mask")
        if legal_mask is not None:
            return self._require_legal_mask(legal_mask, expected_shape=expected_shape)

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            raise ValueError("batch must include either legal_actions, legal_mask, or legal_ids/legal_offsets")
        mask = LegalActionBatch.from_packed(legal_ids, legal_offsets).to_mask(
            expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
            action_space=action_dim,
        )
        return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

    def _resolve_packed_legal_actions(self, batch: Any, *, expected_shape: torch.Size) -> tuple[Tensor, Tensor] | None:
        resolved = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if resolved is None:
            return None
        return resolved[0], resolved[1]

    def _resolve_packed_legal_actions_with_meta(
        self,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        legal_actions = _batch_value(batch, "legal_actions")
        if (
            isinstance(legal_actions, LegalActionBatch)
            and legal_actions.ids is not None
            and legal_actions.offsets is not None
        ):
            ids = torch.as_tensor(legal_actions.ids, device=self._model_parameter().device, dtype=torch.long)
            offsets = torch.as_tensor(legal_actions.offsets, device=self._model_parameter().device, dtype=torch.long)
            expected_rows = int(expected_shape[0] * expected_shape[1])
            if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
                raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
            meta = (
                None
                if legal_actions.meta is None
                else torch.as_tensor(legal_actions.meta, device=self._model_parameter().device, dtype=torch.long)
            )
            if bool(getattr(self.model, "supports_legal_candidate_scoring", False)) and meta is None:
                raise ValueError("structured learner updates require packed legal action metadata")
            return ids, offsets, meta

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            return None
        ids = torch.as_tensor(legal_ids, device=self._model_parameter().device, dtype=torch.long)
        offsets = torch.as_tensor(legal_offsets, device=self._model_parameter().device, dtype=torch.long)
        expected_rows = int(expected_shape[0] * expected_shape[1])
        if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
            raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
        legal_action_meta = _batch_value(batch, "legal_action_meta")
        meta = (
            None
            if legal_action_meta is None
            else torch.as_tensor(legal_action_meta, device=self._model_parameter().device, dtype=torch.long)
        )
        if bool(getattr(self.model, "supports_legal_candidate_scoring", False)) and meta is None:
            raise ValueError("structured learner updates require packed legal action metadata")
        return ids, offsets, meta

    def _packed_legal_action_view(
        self,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
    ) -> Any:
        ids, offsets, meta = packed_legal
        return SimpleNamespace(ids=ids, offsets=offsets, meta=meta)

    def _slice_packed_legal_rows_with_meta(
        self,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        return _slice_packed_legal_rows_with_meta_impl(packed_legal, row_indices)

    def _packed_candidate_positions_for_rows(
        self,
        offsets: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        return _packed_candidate_positions_for_rows_impl(offsets, row_indices)

    def _scatter_packed_candidate_values(
        self,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        row_indices: Tensor,
        subset_values: Tensor,
        *,
        fill_value: float = 0.0,
    ) -> Tensor:
        return _scatter_packed_candidate_values_impl(
            packed_legal,
            row_indices,
            subset_values,
            fill_value=fill_value,
        )

    def _subset_observation_context_rows(
        self,
        observation_context: Mapping[str, Tensor],
        row_indices: Tensor,
        *,
        row_count: int,
    ) -> dict[str, Tensor]:
        return _subset_observation_context_rows_impl(
            observation_context,
            row_indices,
            row_count=row_count,
        )

    def _has_raw_vtrace_inputs(self, batch: Any) -> bool:
        if any(_batch_value(batch, key) is None for key in ("rewards", "discounts", "behavior_logp")):
            return False
        if _batch_value(batch, "bootstrap_value") is not None:
            return True
        return all(
            _batch_value(batch, key) is not None for key in ("bootstrap_obs", "bootstrap_actor", "final_hidden_state")
        )

    def _resolve_vtrace_bootstrap_value(
        self,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor:
        current_bootstrap = self._current_model_bootstrap_value(batch, batch_size=batch_size, like=like)
        if current_bootstrap is not None:
            return current_bootstrap
        bootstrap_value = self._float_input(_batch_value(batch, "bootstrap_value"))
        if bootstrap_value.ndim != 1 or bootstrap_value.shape[0] != batch_size:
            raise ValueError(f"bootstrap_value must have shape ({batch_size},), got {tuple(bootstrap_value.shape)}")
        return bootstrap_value

    def _current_model_bootstrap_value(
        self,
        batch: Any,
        *,
        batch_size: int,
        like: Tensor,
    ) -> Tensor | None:
        bootstrap_obs_value = _batch_value(batch, "bootstrap_obs")
        bootstrap_actor_value = _batch_value(batch, "bootstrap_actor")
        final_hidden_value = _batch_value(batch, "final_hidden_state")
        if bootstrap_obs_value is None or bootstrap_actor_value is None or final_hidden_value is None:
            return None
        if self.model is None:
            return None
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not hasattr(forward_model, "value_seat_aware") and not hasattr(forward_model, "forward_seat_aware"):
            return None
        bootstrap_obs = self._tensor_on_model_device(bootstrap_obs_value, dtype=like.dtype)
        if bootstrap_obs.ndim != 2 or bootstrap_obs.shape[0] != batch_size:
            raise ValueError(
                f"bootstrap_obs must have shape ({batch_size}, observation), got {tuple(bootstrap_obs.shape)}"
            )
        bootstrap_actor = self._optional_batch_seat_field(
            bootstrap_actor_value,
            field_name="bootstrap_actor",
            expected_batch_size=batch_size,
        )
        if bootstrap_actor is None:
            return None
        final_hidden_state = self._tensor_on_model_device(final_hidden_value, dtype=like.dtype)
        if final_hidden_state.ndim != 3:
            return None
        if final_hidden_state.shape[0] != batch_size:
            raise ValueError(
                f"final_hidden_state batch mismatch: expected {batch_size}, got {final_hidden_state.shape[0]}"
            )
        if final_hidden_state.shape[1] != 2:
            raise ValueError(f"final_hidden_state seat mismatch: expected 2, got {final_hidden_state.shape[1]}")
        valid_rows = ((bootstrap_actor == 0) | (bootstrap_actor == 1)).to(dtype=torch.bool)
        bootstrap_value = torch.zeros((batch_size,), dtype=like.dtype, device=like.device)
        if not bool(valid_rows.any().item()):
            return bootstrap_value
        with torch.no_grad():
            value_seat_aware = getattr(forward_model, "value_seat_aware", None)
            if callable(value_seat_aware):
                value_tensor = value_seat_aware(
                    bootstrap_obs[valid_rows],
                    bootstrap_actor[valid_rows],
                    final_hidden_state[valid_rows],
                )
            else:
                _logits_tensor, value_tensor, _next_hidden = forward_model.forward_seat_aware(
                    bootstrap_obs[valid_rows],
                    bootstrap_actor[valid_rows],
                    final_hidden_state[valid_rows],
                )
        bootstrap_value[valid_rows] = torch.as_tensor(
            value_tensor,
            device=bootstrap_value.device,
            dtype=bootstrap_value.dtype,
        )
        return bootstrap_value

    def _float_target(self, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"target must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _optional_batch_seat_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_batch_size: int,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.ndim != 1 or tensor.shape[0] != expected_batch_size:
            raise ValueError(f"{field_name} must have shape ({expected_batch_size},), got {tuple(tensor.shape)}")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{field_name} values must be 0 or 1")
        return tensor

    def _prepare_legacy_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                "initial_hidden_state must be 2D (batch, hidden_size) when to_play_seat/actor is absent, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        return tensor

    def _prepare_seat_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 3:
            raise ValueError(
                "initial_hidden_state must be 3D (batch, seat, hidden_size) when to_play_seat/actor is present, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        if tensor.shape[1] != 2:
            raise ValueError(f"initial_hidden_state seat mismatch: expected 2, got {tensor.shape[1]}")
        return tensor

    def _prepare_acting_seat_batch(
        self,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        seat_tensor = self._optional_time_major_seat_field(
            to_play_seat,
            field_name="to_play_seat",
            expected_shape=expected_shape,
        )
        actor_tensor = self._optional_time_major_seat_field(
            actor,
            field_name="actor",
            expected_shape=expected_shape,
        )

        if seat_tensor is None:
            return actor_tensor
        if actor_tensor is None:
            return seat_tensor
        if not torch.equal(seat_tensor, actor_tensor):
            raise ValueError("actor must match to_play_seat when both are provided")
        return seat_tensor

    def _optional_time_major_seat_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{field_name} values must be 0 or 1")
        return tensor

    def _optional_time_major_loss_mask(
        self,
        value: Any,
        *,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"policy_train_mask must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor.clamp(min=0.0, max=1.0)

    def _optional_time_major_index_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _optional_time_major_bool_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device, dtype=torch.bool)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _float_input(self, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self, value: Any, *, dtype: torch.dtype) -> Tensor:
        if value is None:
            raise ValueError("batch field is required")
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        return tensor.to(dtype=dtype)

    def _model_parameter(self) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter

    def _batch_size(self, batch: Any) -> int:
        return self._numeric_fault_reporter().batch_size(batch)

    def _fault_dir_path(self) -> Path:
        return self._numeric_fault_reporter().fault_dir_path()

    def _batch_fault_snapshot(self, batch: Any) -> dict[str, Any]:
        return self._numeric_fault_reporter().batch_fault_snapshot(batch)

    def _numeric_fault_reporter(self) -> LearnerNumericFaultReporter:
        return LearnerNumericFaultReporter(
            fault_dir=self.fault_dir,
            checkpoint_dir=self.checkpoint_dir,
            logs_dir=self.logs_dir,
            update_count=int(self.update_count),
            policy_version=int(self.policy_version),
            pass_action_id=self.pass_action_id,
        )

    def _write_numeric_fault_bundle(self, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return self._numeric_fault_reporter().write_numeric_fault_bundle(stage=stage, batch=batch, context=context)

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        self._numeric_fault_reporter().ensure_finite_tensor(name, tensor, batch=batch, context=context)

    def _collect_nonfinite_gradients(self, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        return collect_nonfinite_gradients(self.model, grad_norm)

    def _ensure_finite_gradients(self, *, batch: Any, context: dict[str, Any], grad_norm: Tensor) -> None:
        bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
        if not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item()):
            return

        self._raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
        )

    def _raise_for_nonfinite_gradients(
        self,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:

        self._numeric_fault_reporter().raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
        )
