"""Paired-outcome preference replay optimizer-update orchestration."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.scoped_optimizer_update import (
    ScopedOptimizerUpdateSpec,
    run_scoped_impala_optimizer_update,
)


def run_impala_paired_outcome_preference_optimizer_update(
    *,
    learner: Any,
    batch: Any,
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
) -> dict[str, float]:
    def build_loss() -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        return learner._paired_outcome_preference_loss_and_metrics(
            batch,
            beta=float(beta),
            coef=float(coef),
            aggregation=str(aggregation),
            group_balance=bool(group_balance),
            retention_coef=float(retention_coef),
            retention_margin=float(retention_margin),
            retention_role=str(retention_role),
            retention_reference_top_only=bool(retention_reference_top_only),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
            top_action_retention_role=str(top_action_retention_role),
            top_action_retention_reference_top_only=bool(top_action_retention_reference_top_only),
        )

    return run_scoped_impala_optimizer_update(
        learner=learner,
        batch=batch,
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message=("ImpalaLearner requires a model to run a paired outcome preference optimizer step"),
            loss_timer_name="learner_paired_outcome_preference_loss_and_metrics",
        ),
        build_loss=build_loss,
    )


__all__ = ["run_impala_paired_outcome_preference_optimizer_update"]
