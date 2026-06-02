"""IMPALA learner orchestration for structured-teacher auxiliary losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala.structured_summary import (
    ImpalaStructuredSummaryRequest,
    compute_impala_structured_policy_summary,
)
from weiss_rl.learners.impala.teacher_auxiliary_request import compute_impala_teacher_auxiliary
from weiss_rl.learners.impala.teacher_target_inputs import prepare_impala_teacher_target_inputs


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


class ImpalaStructuredTeacherAuxiliaryMixin:
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
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        teacher_target_inputs = prepare_impala_teacher_target_inputs(
            learner=self,
            batch=batch,
            forward_model=forward_model,
            obs=obs,
            logits=logits,
            packed_logits=packed_logits,
            packed_legal=packed_legal,
            loss_mask=loss_mask,
            factorized_result=factorized_result,
            forward_observation_context=forward_observation_context,
            need_packed_view=True,
            teacher_aux_enabled=True,
        )
        packed_view = teacher_target_inputs.packed_view
        teacher_aux_packed_view = teacher_target_inputs.teacher_aux_packed_view
        public_heuristic_target_logits = teacher_target_inputs.public_heuristic_target_logits

        teacher_aux_result = compute_impala_teacher_auxiliary(
            learner=self,
            batch=batch,
            logits=logits,
            legal_mask=legal_mask,
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            expected_shape=values.shape,
            packed_legal=packed_legal,
            packed_view=teacher_aux_packed_view,
            factorized_result=factorized_result,
            public_heuristic_target_logits=public_heuristic_target_logits,
            batch_value=_batch_value,
        )
        teacher_aux_loss = teacher_aux_result.loss
        teacher_metrics = teacher_aux_result.metrics
        context: dict[str, Any] = {
            "auxiliary_loss": teacher_aux_loss.detach(),
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
            "policy_train_mask": loss_mask.detach(),
            **teacher_aux_result.context,
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
            metrics.update(
                compute_impala_structured_policy_summary(
                    ImpalaStructuredSummaryRequest(
                        logits=logits,
                        legal_mask=legal_mask,
                        action_catalog=action_catalog,
                        packed_legal=packed_legal,
                        packed_view=packed_view,
                        factorized_result=factorized_result,
                    ),
                    record_timing_ms=self._record_timing_ms,
                )
            )
        return teacher_aux_loss, metrics, context


__all__ = ["ImpalaStructuredTeacherAuxiliaryMixin"]
