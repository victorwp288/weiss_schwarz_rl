"""IMPALA learner orchestration for paired-outcome preference losses."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.paired_auxiliary_batch import (
    batch_value,
    resolve_paired_auxiliary_batch_inputs,
    resolve_paired_auxiliary_reset_before_step,
)
from weiss_rl.learners.impala.paired_outcome_candidates import compute_paired_outcome_candidate_logps
from weiss_rl.learners.impala.paired_outcome_outputs import (
    build_paired_outcome_preference_context,
    build_paired_outcome_preference_metrics,
)
from weiss_rl.learners.paired_outcome_preference.loss import paired_outcome_preference_loss


class ImpalaPairedOutcomeAuxiliaryMixin:
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

        inputs = resolve_paired_auxiliary_batch_inputs(
            self,
            batch,
            packed_legal_error="paired outcome preference replay requires packed legal_ids/legal_offsets",
        )
        obs = inputs.obs
        expected_shape = inputs.expected_shape
        packed_legal = inputs.packed_legal
        loss_mask = inputs.loss_mask
        actions = self._require_actions(batch_value(batch, "actions"), expected_shape=expected_shape)
        preference_pair_ids = self._optional_time_major_index_field(
            batch_value(batch, "preference_pair_id"),
            field_name="preference_pair_id",
            expected_shape=expected_shape,
        )
        if preference_pair_ids is None:
            raise ValueError("paired outcome preference replay requires batch.preference_pair_id")
        preference_role = self._optional_time_major_index_field(
            batch_value(batch, "preference_role"),
            field_name="preference_role",
            expected_shape=expected_shape,
        )
        if preference_role is None:
            raise ValueError("paired outcome preference replay requires batch.preference_role")
        preference_group_ids = self._optional_time_major_index_field(
            batch_value(batch, "preference_group_id"),
            field_name="preference_group_id",
            expected_shape=expected_shape,
        )
        if group_balance and preference_group_ids is None:
            raise ValueError("paired outcome preference group balance requires batch.preference_group_id")
        preference_pair_weights = self._optional_time_major_float_field(
            batch_value(batch, "preference_pair_weight"),
            field_name="preference_pair_weight",
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        preference_retention_mask = self._optional_time_major_loss_mask(
            batch_value(batch, "preference_retention_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        preference_top_action_retention_mask = self._optional_time_major_loss_mask(
            batch_value(batch, "preference_top_action_retention_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        reset_before_step = resolve_paired_auxiliary_reset_before_step(
            self,
            batch,
            expected_shape=expected_shape,
        )

        candidate_logps = compute_paired_outcome_candidate_logps(
            self,
            batch,
            obs=obs,
            packed_legal=packed_legal,
            actions=actions,
            reset_before_step=reset_before_step,
        )

        base_loss, preference_metrics, preference_context = paired_outcome_preference_loss(
            current_action_logp=candidate_logps.current_action_logp,
            reference_action_logp=candidate_logps.reference_action_logp,
            current_best_non_target_logp=candidate_logps.current_best_non_target_logp,
            reference_best_non_target_logp=candidate_logps.reference_best_non_target_logp,
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
        context = build_paired_outcome_preference_context(
            weighted_loss=weighted_loss,
            loss_mask=loss_mask,
            candidate_logps=candidate_logps,
            preference_context=preference_context,
        )
        self._ensure_finite_tensor("paired_outcome_preference_loss", weighted_loss, batch=batch, context=context)
        metrics = build_paired_outcome_preference_metrics(
            weighted_loss=weighted_loss,
            coef=float(coef),
            beta=float(beta),
            aggregation=aggregation,
            group_balance=bool(group_balance),
            retention_coef=float(retention_coef),
            retention_margin=float(retention_margin),
            retention_reference_top_only=bool(retention_reference_top_only),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
            top_action_retention_reference_top_only=bool(top_action_retention_reference_top_only),
            preference_metrics=preference_metrics,
        )
        zero = candidate_logps.current_candidate_log_probs.sum() * 0.0
        return weighted_loss if float(coef) != 0.0 else zero, metrics, context


__all__ = ["ImpalaPairedOutcomeAuxiliaryMixin"]
