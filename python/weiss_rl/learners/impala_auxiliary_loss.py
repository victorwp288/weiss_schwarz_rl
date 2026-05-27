"""Auxiliary structured-teacher loss mixin for the IMPALA learner."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.action_logp import packed_selected_action_logp
from weiss_rl.learners.paired_outcome_preference_loss import paired_outcome_preference_loss
from weiss_rl.learners.paired_swing_loss import (
    packed_paired_swing_margin_loss,
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
)
from weiss_rl.learners.structured_auxiliary import packed_structured_legal_view
from weiss_rl.learners.structured_policy_metrics import summarize_structured_policy_metrics
from weiss_rl.learners.structured_teacher_auxiliary import compute_structured_teacher_auxiliary_metrics
from weiss_rl.learners.tensor_ops import segment_max


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _packed_best_non_target_logp(
    candidate_log_probs: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    actions: Tensor,
) -> Tensor:
    offsets = packed_offsets.to(device=candidate_log_probs.device, dtype=torch.long)
    ids = packed_ids.to(device=candidate_log_probs.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    lengths = offsets[1:] - offsets[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, dtype=torch.long, device=candidate_log_probs.device),
        lengths,
    )
    if int(row_indices.numel()) != int(candidate_log_probs.numel()):
        raise ValueError("packed offsets do not match packed candidate log-probs")
    flat_actions = actions.reshape(-1).to(device=candidate_log_probs.device, dtype=torch.long)
    if int(flat_actions.numel()) != row_count:
        raise ValueError(f"actions row count {int(flat_actions.numel())} does not match packed row count {row_count}")
    row_targets = flat_actions.index_select(0, row_indices)
    non_target_scores = torch.where(
        ids != row_targets,
        candidate_log_probs.to(dtype=torch.float32),
        torch.full_like(candidate_log_probs.to(dtype=torch.float32), -torch.inf),
    )
    return segment_max(non_target_scores, row_indices, row_count)


class ImpalaAuxiliaryLossMixin:
    model: Any
    compiled_model: Any
    teacher_family_coef: float
    teacher_slot_coef: float
    teacher_hand_coef: float
    teacher_move_source_coef: float
    teacher_attack_type_coef: float
    teacher_action_coef: float
    teacher_same_family_action_coef: float
    teacher_action_margin_coef: float
    teacher_action_margin: float
    teacher_same_family_action_margin_coef: float
    teacher_same_family_action_margin: float
    teacher_exact_action_families: tuple[str, ...]
    teacher_public_heuristic_coef: float
    teacher_public_heuristic_temperature: float
    teacher_public_nonpass_over_pass_coef: float
    teacher_public_nonpass_over_pass_margin: float
    teacher_public_heuristic_families: tuple[str, ...]

    def _auxiliary_loss_and_metrics(self: Any, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute auxiliary losses")
        action_catalog = getattr(self.model, "action_catalog", None)
        if not isinstance(action_catalog, ActionCatalog):
            raise ValueError("structured auxiliary pretraining requires a structured action catalog")
        if not self._teacher_aux_active(auxiliary_update=True):
            zero = self._model_parameter().sum() * 0.0
            return zero, {"loss": 0.0, "policy_train_fraction": 0.0}, {}

        obs = self._require_obs(_batch_value(batch, "obs"))
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        factorized_result = None
        forward_observation_context: Mapping[str, Tensor] | None = None
        if self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            factorized_result, packed_legal = self._evaluate_factorized_time_major(
                batch,
                obs=obs,
                actions=None,
            )
            logits = None
            packed_logits = None
            values = factorized_result.values
        else:
            forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=_batch_value(batch, "legal_actions"),
                opponent_context_index=_batch_value(batch, "opponent_context_index"),
            )
            logits = forward.logits
            packed_logits = forward.packed_logits
            values = forward.values
            forward_observation_context = forward.observation_context
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=True)
        packed_view = None
        if packed_legal is not None and factorized_result is None:
            packed_view_started = time.perf_counter()
            packed_view = packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
        teacher_aux_packed_view = packed_view
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        public_heuristic_target_logits = None
        public_candidate_target_active = (
            float(self.teacher_public_heuristic_coef) != 0.0 or float(self.teacher_public_nonpass_over_pass_coef) != 0.0
        )
        factorized_candidate_teacher_view_active = factorized_result is not None and (
            public_candidate_target_active
            or float(self.teacher_action_margin_coef) != 0.0
            or float(self.teacher_same_family_action_margin_coef) != 0.0
        )
        if (
            packed_legal is not None
            and (public_candidate_target_active or factorized_candidate_teacher_view_active)
            and (factorized_result is not None or hasattr(forward_model, "score_packed_public_heuristic_candidates"))
        ):
            if factorized_result is not None:
                teacher_aux_packed_view, public_heuristic_target_logits = (
                    self._factorized_public_heuristic_teacher_view(
                        batch,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        score_public_target=public_candidate_target_active,
                    )
                )
            elif public_candidate_target_active:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    public_heuristic_target_logits = self._packed_public_heuristic_target_logits(
                        forward_model=forward_model,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        observation_context=forward_observation_context,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)

        teacher_aux_started = time.perf_counter()
        teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
            logits=logits,
            legal_mask=legal_mask,
            teacher_family=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_family"),
                field_name="teacher_family",
                expected_shape=values.shape,
            ),
            teacher_slot=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_slot"),
                field_name="teacher_slot",
                expected_shape=values.shape,
            ),
            teacher_move_source=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_move_source"),
                field_name="teacher_move_source",
                expected_shape=values.shape,
            ),
            teacher_attack_type=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_attack_type"),
                field_name="teacher_attack_type",
                expected_shape=values.shape,
            ),
            teacher_action=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_action"),
                field_name="teacher_action",
                expected_shape=values.shape,
            ),
            teacher_valid=self._optional_time_major_bool_field(
                _batch_value(batch, "teacher_valid"),
                field_name="teacher_valid",
                expected_shape=values.shape,
            ),
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            family_coef=float(self.teacher_family_coef),
            slot_coef=float(self.teacher_slot_coef),
            hand_coef=float(self.teacher_hand_coef),
            move_source_coef=float(self.teacher_move_source_coef),
            attack_type_coef=float(self.teacher_attack_type_coef),
            action_coef=float(self.teacher_action_coef),
            same_family_action_coef=float(self.teacher_same_family_action_coef),
            action_margin_coef=float(self.teacher_action_margin_coef),
            action_margin=float(self.teacher_action_margin),
            same_family_action_margin_coef=float(self.teacher_same_family_action_margin_coef),
            same_family_action_margin=float(self.teacher_same_family_action_margin),
            exact_action_families=tuple(self.teacher_exact_action_families),
            public_heuristic_coef=float(self.teacher_public_heuristic_coef),
            public_heuristic_temperature=float(self.teacher_public_heuristic_temperature),
            public_nonpass_over_pass_coef=float(self.teacher_public_nonpass_over_pass_coef),
            public_nonpass_over_pass_margin=float(self.teacher_public_nonpass_over_pass_margin),
            public_heuristic_families=tuple(self.teacher_public_heuristic_families),
            public_heuristic_target_logits=public_heuristic_target_logits,
            packed_ids=None if packed_legal is None else packed_legal[0],
            packed_offsets=None if packed_legal is None else packed_legal[1],
            packed_meta=None if packed_legal is None else packed_legal[2],
            packed_view=teacher_aux_packed_view,
            factorized_family_log_probs=None if factorized_result is None else factorized_result.family_log_probs,
            factorized_play_slot_log_probs=None if factorized_result is None else factorized_result.play_slot_log_probs,
            factorized_move_source_log_probs=None
            if factorized_result is None
            else getattr(factorized_result, "move_source_log_probs", None),
            factorized_move_slot_log_probs=None if factorized_result is None else factorized_result.move_slot_log_probs,
            factorized_attack_slot_log_probs=None
            if factorized_result is None
            else factorized_result.attack_slot_log_probs,
            factorized_attack_type_log_probs=None
            if factorized_result is None
            else factorized_result.attack_type_log_probs,
            factorized_top_action_ids=None
            if factorized_result is None
            else getattr(factorized_result, "top_action_ids", None),
            factorized_same_family_action_logp=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_action_logp", None),
            factorized_same_family_top_action_ids=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_top_action_ids", None),
            factorized_same_family_arg0_logp=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_arg0_logp", None),
            factorized_same_family_top_arg0=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_top_arg0", None),
        )
        self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
        context: dict[str, Any] = {
            "auxiliary_loss": teacher_aux_loss.detach(),
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
            "policy_train_mask": loss_mask.detach(),
            **teacher_context,
        }
        if factorized_result is not None:
            context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
        self._ensure_finite_tensor("auxiliary_loss", teacher_aux_loss, batch=batch, context=context)
        metrics = {
            "loss": float(teacher_aux_loss.detach().item()),
            "policy_train_fraction": float(loss_mask.mean().detach().item()),
        }
        metrics.update(teacher_metrics)
        if emit_structured_metrics:
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                    factorized_family_log_probs=None
                    if factorized_result is None
                    else factorized_result.family_log_probs,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return teacher_aux_loss, metrics, context

    def _paired_outcome_preference_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        beta: float,
        coef: float,
        aggregation: str = "mean",
        group_balance: bool = False,
        retention_coef: float = 0.0,
        retention_margin: float = 0.0,
        retention_role: str = "preferred",
        retention_reference_top_only: bool = False,
        top_action_retention_coef: float = 0.0,
        top_action_retention_margin: float = 0.0,
        top_action_retention_role: str = "all",
        top_action_retention_reference_top_only: bool = False,
    ) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute paired outcome preference losses")

        obs = self._require_obs(_batch_value(batch, "obs"))
        expected_shape = obs.shape[:2]
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if packed_legal is None:
            raise ValueError("paired outcome preference replay requires packed legal_ids/legal_offsets")
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=expected_shape)
        preference_pair_ids = self._optional_time_major_index_field(
            _batch_value(batch, "preference_pair_id"),
            field_name="preference_pair_id",
            expected_shape=expected_shape,
        )
        if preference_pair_ids is None:
            raise ValueError("paired outcome preference replay requires batch.preference_pair_id")
        preference_role = self._optional_time_major_index_field(
            _batch_value(batch, "preference_role"),
            field_name="preference_role",
            expected_shape=expected_shape,
        )
        if preference_role is None:
            raise ValueError("paired outcome preference replay requires batch.preference_role")
        preference_group_ids = self._optional_time_major_index_field(
            _batch_value(batch, "preference_group_id"),
            field_name="preference_group_id",
            expected_shape=expected_shape,
        )
        if group_balance and preference_group_ids is None:
            raise ValueError("paired outcome preference group balance requires batch.preference_group_id")
        preference_pair_weights = self._optional_time_major_float_field(
            _batch_value(batch, "preference_pair_weight"),
            field_name="preference_pair_weight",
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        preference_retention_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "preference_retention_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        preference_top_action_retention_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "preference_top_action_retention_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        if loss_mask is None:
            loss_mask = torch.ones(expected_shape, device=obs.device, dtype=obs.dtype)
        reset_before_step = self._optional_time_major_bool_field(
            _batch_value(batch, "reset_before_step"),
            field_name="reset_before_step",
            expected_shape=expected_shape,
        )

        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            raise ValueError("paired outcome preference replay currently requires the factorized packed learner path")
        anchor_model = self._ensure_policy_anchor_model()
        if not self._should_use_factorized_legal_policy(anchor_model, packed_legal=packed_legal):
            raise ValueError("paired outcome preference replay requires a factorized structured reference model")

        current_candidate_log_probs = self._factorized_candidate_log_probs_for_model(
            forward_model,
            batch,
            obs=obs,
            packed_legal=packed_legal,
            reset_before_step=reset_before_step,
        )
        with torch.no_grad():
            reference_candidate_log_probs = self._factorized_candidate_log_probs_for_model(
                anchor_model,
                batch,
                obs=obs,
                packed_legal=packed_legal,
                reset_before_step=reset_before_step,
            )
        current_action_logp = packed_selected_action_logp(
            current_candidate_log_probs,
            packed_legal[0],
            packed_legal[1],
            actions,
            pass_action_id=self.pass_action_id,
            strict=False,
        ).reshape(expected_shape)
        current_best_non_target_logp = _packed_best_non_target_logp(
            current_candidate_log_probs,
            packed_legal[0],
            packed_legal[1],
            actions,
        ).reshape(expected_shape)
        reference_action_logp = packed_selected_action_logp(
            reference_candidate_log_probs,
            packed_legal[0],
            packed_legal[1],
            actions,
            pass_action_id=self.pass_action_id,
            strict=False,
        ).reshape(expected_shape)
        reference_best_non_target_logp = _packed_best_non_target_logp(
            reference_candidate_log_probs,
            packed_legal[0],
            packed_legal[1],
            actions,
        ).reshape(expected_shape)

        base_loss, preference_metrics, preference_context = paired_outcome_preference_loss(
            current_action_logp=current_action_logp,
            reference_action_logp=reference_action_logp,
            current_best_non_target_logp=current_best_non_target_logp,
            reference_best_non_target_logp=reference_best_non_target_logp,
            preference_pair_ids=preference_pair_ids,
            preference_role=preference_role,
            preference_group_ids=preference_group_ids,
            preference_pair_weights=preference_pair_weights,
            loss_mask=loss_mask > 0.0,
            beta=float(beta),
            aggregation=str(aggregation),
            group_balance=bool(group_balance),
            retention_coef=float(retention_coef),
            retention_margin=float(retention_margin),
            retention_role=str(retention_role),
            retention_reference_top_only=bool(retention_reference_top_only),
            retention_scope_mask=None if preference_retention_mask is None else preference_retention_mask > 0.0,
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
            top_action_retention_role=str(top_action_retention_role),
            top_action_retention_reference_top_only=bool(top_action_retention_reference_top_only),
            top_action_retention_scope_mask=None
            if preference_top_action_retention_mask is None
            else preference_top_action_retention_mask > 0.0,
        )
        weighted_loss = base_loss * float(coef)
        context: dict[str, Any] = {
            "paired_outcome_preference_loss": weighted_loss.detach(),
            "policy_train_mask": loss_mask.detach(),
            "current_action_logp": current_action_logp.detach(),
            "current_best_non_target_logp": current_best_non_target_logp.detach(),
            "reference_action_logp": reference_action_logp.detach(),
            "reference_best_non_target_logp": reference_best_non_target_logp.detach(),
            **preference_context,
        }
        self._ensure_finite_tensor("paired_outcome_preference_loss", weighted_loss, batch=batch, context=context)
        metrics = {
            "loss": float(weighted_loss.detach().item()),
            "paired_outcome_preference_weighted_loss": float(weighted_loss.detach().item()),
            "paired_outcome_preference_coef": float(coef),
            "paired_outcome_preference_beta": float(beta),
            "paired_outcome_preference_aggregation_sum": 1.0 if str(aggregation).strip().lower() == "sum" else 0.0,
            "paired_outcome_preference_group_balance": 1.0 if group_balance else 0.0,
            "paired_outcome_preference_retention_coef": float(retention_coef),
            "paired_outcome_preference_retention_margin": float(retention_margin),
            "paired_outcome_preference_retention_reference_top_only": 1.0 if retention_reference_top_only else 0.0,
            "paired_outcome_preference_top_action_retention_coef": float(top_action_retention_coef),
            "paired_outcome_preference_top_action_retention_margin": float(top_action_retention_margin),
            "paired_outcome_preference_top_action_retention_reference_top_only": 1.0
            if top_action_retention_reference_top_only
            else 0.0,
        }
        metrics.update(preference_metrics)
        zero = current_candidate_log_probs.sum() * 0.0
        return weighted_loss if float(coef) != 0.0 else zero, metrics, context

    def _paired_swing_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        margin: float,
        coef: float,
        positive_action_source: str,
        negative_action_source: str,
        loss_scope: str = "row",
        compare_to: str = "negative",
        margin_retention_coef: float = 0.0,
        margin_retention_margin: float = 0.0,
        top_action_retention_coef: float = 0.0,
        top_action_retention_margin: float = 0.0,
    ) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute paired-swing losses")

        obs = self._require_obs(_batch_value(batch, "obs"))
        expected_shape = obs.shape[:2]
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if packed_legal is None:
            raise ValueError("paired-swing replay requires packed legal_ids/legal_offsets")

        positive_actions = self._paired_swing_action_tensor(
            batch,
            source=positive_action_source,
            expected_shape=expected_shape,
        )
        negative_actions = self._paired_swing_action_tensor(
            batch,
            source=negative_action_source,
            expected_shape=expected_shape,
        )
        negative_valid = self._optional_time_major_bool_field(
            _batch_value(batch, "teacher_valid"),
            field_name="teacher_valid",
            expected_shape=expected_shape,
        )
        if negative_valid is None:
            negative_valid = torch.ones(expected_shape, device=obs.device, dtype=torch.bool)
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        if loss_mask is None:
            loss_mask = torch.ones(expected_shape, device=obs.device, dtype=obs.dtype)
        source_label_id = self._optional_time_major_index_field(
            _batch_value(batch, "source_label_id"),
            field_name="source_label_id",
            expected_shape=expected_shape,
        )

        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        factorized_result = None
        logits = None
        packed_logits = None
        values = None
        reference_packed_logits = None
        if self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            factorized_view, _target_logits = self._factorized_public_heuristic_teacher_view(
                batch,
                obs=obs,
                loss_mask=loss_mask,
                packed_legal=packed_legal,
                score_public_target=False,
                reattach_initial_hidden_context_gradient=True,
            )
            if factorized_view is None:
                raise ValueError("factorized paired-swing replay could not build a packed candidate view")
            packed_view = factorized_view
            zero = packed_view.logits.sum() * 0.0
            if float(margin_retention_coef) != 0.0 or float(top_action_retention_coef) != 0.0:
                anchor_model = self._ensure_policy_anchor_model()
                with torch.no_grad():
                    reference_packed_logits = self._factorized_candidate_log_probs_for_model(
                        anchor_model,
                        batch,
                        obs=obs,
                        packed_legal=packed_legal,
                        reset_before_step=self._optional_time_major_bool_field(
                            _batch_value(batch, "reset_before_step"),
                            field_name="reset_before_step",
                            expected_shape=expected_shape,
                        ),
                    )
        else:
            if float(margin_retention_coef) != 0.0 or float(top_action_retention_coef) != 0.0:
                raise ValueError("paired-swing retention currently requires the factorized packed learner path")
            legal_actions = _batch_value(batch, "legal_actions")
            if legal_actions is None:
                legal_actions = self._packed_legal_action_view(packed_legal)
            forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=legal_actions,
                policy_train_mask=loss_mask,
                reset_before_step=_batch_value(batch, "reset_before_step"),
                opponent_context_index=_batch_value(batch, "opponent_context_index"),
            )
            logits = forward.logits
            packed_logits = forward.packed_logits
            values = forward.values
            packed_view = packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            if packed_view is None:
                raise ValueError("paired-swing replay requires packed legal action metadata")
            zero = values.sum() * 0.0

        base_loss, swing_metrics, swing_context = packed_paired_swing_margin_loss(
            packed_logits=packed_view.logits,
            legal_ids=packed_legal[0],
            legal_offsets=packed_legal[1],
            positive_actions=positive_actions,
            negative_actions=negative_actions,
            negative_valid=negative_valid,
            loss_mask=loss_mask,
            margin=float(margin),
            pass_action_id=self.pass_action_id,
            loss_scope=str(loss_scope),
            compare_to=str(compare_to),
            group_ids=source_label_id,
            reference_packed_logits=reference_packed_logits,
            margin_retention_coef=float(margin_retention_coef),
            margin_retention_margin=float(margin_retention_margin),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
        )
        weighted_loss = base_loss * float(coef)
        context: dict[str, Any] = {
            "paired_swing_loss": weighted_loss.detach(),
            "logits": None if logits is None else logits.detach(),
            "packed_logits": packed_view.logits.detach(),
            "values": None if values is None else values.detach(),
            "policy_train_mask": loss_mask.detach(),
            **swing_context,
        }
        if factorized_result is not None:
            context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
        self._ensure_finite_tensor("paired_swing_loss", weighted_loss, batch=batch, context=context)
        metrics = {
            "loss": float(weighted_loss.detach().item()),
            "paired_swing_weighted_loss": float(weighted_loss.detach().item()),
            "paired_swing_coef": float(coef),
            "paired_swing_margin": float(margin),
            "paired_swing_margin_retention_coef": float(margin_retention_coef),
            "paired_swing_margin_retention_margin": float(margin_retention_margin),
            "paired_swing_top_action_retention_coef": float(top_action_retention_coef),
            "paired_swing_top_action_retention_margin": float(top_action_retention_margin),
            "paired_swing_positive_action_source_teacher": 1.0 if positive_action_source == "teacher_action" else 0.0,
            "paired_swing_negative_action_source_teacher": 1.0 if negative_action_source == "teacher_action" else 0.0,
            "paired_swing_loss_scope_episode_mean": 1.0 if str(loss_scope) == "episode_mean" else 0.0,
            "paired_swing_loss_scope_label_mean": 1.0 if str(loss_scope) == "label_mean" else 0.0,
            "paired_swing_compare_to_top_other": 1.0 if str(compare_to).strip().lower() == "top_other" else 0.0,
        }
        metrics.update(swing_metrics)
        return weighted_loss if float(coef) != 0.0 else zero, metrics, context

    def _paired_swing_full_surface_top_action_retention_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        coef: float,
        margin: float,
        mode: str = "reference_top",
    ) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute full-surface paired-swing retention")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"reference_top", "target_action"}:
            raise ValueError("full-surface paired-swing retention mode must be one of: reference_top, target_action")

        obs = self._require_obs(_batch_value(batch, "obs"))
        expected_shape = obs.shape[:2]
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if packed_legal is None:
            raise ValueError("full-surface paired-swing retention requires packed legal_ids/legal_offsets")
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        if loss_mask is None:
            loss_mask = torch.ones(expected_shape, device=obs.device, dtype=obs.dtype)

        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            raise ValueError("full-surface paired-swing retention requires the factorized packed learner path")
        factorized_view, _target_logits = self._factorized_public_heuristic_teacher_view(
            batch,
            obs=obs,
            loss_mask=loss_mask,
            packed_legal=packed_legal,
            score_public_target=False,
            reattach_initial_hidden_context_gradient=True,
        )
        if factorized_view is None:
            raise ValueError("full-surface paired-swing retention could not build a packed candidate view")
        reference_packed_logits = None
        if normalized_mode == "reference_top":
            anchor_model = self._ensure_policy_anchor_model()
            with torch.no_grad():
                reference_packed_logits = self._factorized_candidate_log_probs_for_model(
                    anchor_model,
                    batch,
                    obs=obs,
                    packed_legal=packed_legal,
                    reset_before_step=self._optional_time_major_bool_field(
                        _batch_value(batch, "reset_before_step"),
                        field_name="reset_before_step",
                        expected_shape=expected_shape,
                    ),
                )
            base_loss, retention_metrics, retention_context = packed_top_action_retention_loss(
                packed_logits=factorized_view.logits,
                reference_packed_logits=reference_packed_logits,
                legal_ids=packed_legal[0],
                legal_offsets=packed_legal[1],
                loss_mask=loss_mask,
                retention_margin=float(margin),
                metric_prefix="paired_swing_full_surface",
            )
        else:
            target_actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=expected_shape)
            target_valid = target_actions >= 0
            base_loss, retention_metrics, retention_context = packed_target_action_retention_loss(
                packed_logits=factorized_view.logits,
                legal_ids=packed_legal[0],
                legal_offsets=packed_legal[1],
                target_actions=target_actions,
                target_valid=target_valid,
                loss_mask=loss_mask,
                retention_margin=float(margin),
                metric_prefix="paired_swing_full_surface_target",
            )
        weighted_loss = base_loss * float(coef)
        context: dict[str, Any] = {
            "paired_swing_full_surface_top_action_retention_loss": weighted_loss.detach(),
            "paired_swing_full_surface_packed_logits": factorized_view.logits.detach(),
            "policy_train_mask": loss_mask.detach(),
            **retention_context,
        }
        self._ensure_finite_tensor(
            "paired_swing_full_surface_top_action_retention_loss",
            weighted_loss,
            batch=batch,
            context=context,
        )
        metrics = {
            "paired_swing_full_surface_top_action_retention_weighted_loss": float(weighted_loss.detach().item()),
            "paired_swing_full_surface_top_action_retention_coef": float(coef),
            "paired_swing_full_surface_top_action_retention_margin": float(margin),
            "paired_swing_full_surface_top_action_retention_mode_reference_top": 1.0
            if normalized_mode == "reference_top"
            else 0.0,
            "paired_swing_full_surface_top_action_retention_mode_target_action": 1.0
            if normalized_mode == "target_action"
            else 0.0,
        }
        metrics.update(retention_metrics)
        zero = factorized_view.logits.sum() * 0.0
        return weighted_loss if float(coef) != 0.0 else zero, metrics, context

    def _paired_swing_action_tensor(self: Any, batch: Any, *, source: str, expected_shape: torch.Size) -> Tensor:
        normalized = str(source).strip().lower()
        if normalized == "actions":
            return self._require_actions(_batch_value(batch, "actions"), expected_shape=expected_shape)
        if normalized == "teacher_action":
            teacher_action = self._optional_time_major_index_field(
                _batch_value(batch, "teacher_action"),
                field_name="teacher_action",
                expected_shape=expected_shape,
            )
            if teacher_action is None:
                raise ValueError("paired-swing replay source teacher_action requires batch.teacher_action")
            return teacher_action
        raise ValueError("paired-swing action source must be one of: actions, teacher_action")
