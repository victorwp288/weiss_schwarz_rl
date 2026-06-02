"""Context and metric assembly for IMPALA paired-outcome preference replay."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.paired_outcome_candidates import PairedOutcomeCandidateLogps


def build_paired_outcome_preference_context(
    *,
    weighted_loss: Tensor,
    loss_mask: Tensor,
    candidate_logps: PairedOutcomeCandidateLogps,
    preference_context: dict[str, Tensor],
) -> dict[str, Any]:
    return {
        "paired_outcome_preference_loss": weighted_loss.detach(),
        "policy_train_mask": loss_mask.detach(),
        "current_action_logp": candidate_logps.current_action_logp.detach(),
        "current_best_non_target_logp": candidate_logps.current_best_non_target_logp.detach(),
        "reference_action_logp": candidate_logps.reference_action_logp.detach(),
        "reference_best_non_target_logp": candidate_logps.reference_best_non_target_logp.detach(),
        **preference_context,
    }


def build_paired_outcome_preference_metrics(
    *,
    weighted_loss: Tensor,
    coef: float,
    beta: float,
    aggregation: str,
    group_balance: bool,
    retention_coef: float,
    retention_margin: float,
    retention_reference_top_only: bool,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    top_action_retention_reference_top_only: bool,
    preference_metrics: dict[str, float],
) -> dict[str, float]:
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
    return metrics


__all__ = ["build_paired_outcome_preference_context", "build_paired_outcome_preference_metrics"]
