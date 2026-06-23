"""Optimizer-step orchestration for :class:`weiss_rl.learners.impala.ImpalaLearner`."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.auxiliary_update import run_impala_auxiliary_optimizer_update
from weiss_rl.learners.impala.normal_update import run_impala_normal_update
from weiss_rl.learners.impala.paired_outcome_update import run_impala_paired_outcome_preference_optimizer_update
from weiss_rl.learners.impala.paired_swing_update import run_impala_paired_swing_optimizer_update
from weiss_rl.learners.impala.scoped_optimizer_update import (
    ScopedOptimizerUpdateSpec,
    run_scoped_impala_optimizer_update,
)


class ImpalaUpdateLoopMixin:
    def update(self: Any, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
        return run_impala_normal_update(learner=self, batch=batch)

    def auxiliary_update(self: Any, batch: Any) -> dict[str, float]:
        """Run one optimizer step using only structured teacher supervision."""
        return run_impala_auxiliary_optimizer_update(learner=self, batch=batch)

    def paired_swing_update(
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
        full_surface_retention_batch: Any | None = None,
        full_surface_top_action_retention_coef: float = 0.0,
        full_surface_top_action_retention_margin: float = 0.0,
        full_surface_top_action_retention_mode: str = "reference_top",
    ) -> dict[str, float]:
        """Run one optimizer step using paired action-margin replay supervision."""
        return run_impala_paired_swing_optimizer_update(
            learner=self,
            batch=batch,
            margin=margin,
            coef=coef,
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
            loss_scope=loss_scope,
            compare_to=compare_to,
            margin_retention_coef=margin_retention_coef,
            margin_retention_margin=margin_retention_margin,
            top_action_retention_coef=top_action_retention_coef,
            top_action_retention_margin=top_action_retention_margin,
            full_surface_retention_batch=full_surface_retention_batch,
            full_surface_top_action_retention_coef=full_surface_top_action_retention_coef,
            full_surface_top_action_retention_margin=full_surface_top_action_retention_margin,
            full_surface_top_action_retention_mode=full_surface_top_action_retention_mode,
        )

    def paired_outcome_preference_update(
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
    ) -> dict[str, float]:
        """Run one optimizer step using paired trajectory/span preference replay."""
        return run_impala_paired_outcome_preference_optimizer_update(
            learner=self,
            batch=batch,
            beta=beta,
            coef=coef,
            aggregation=aggregation,
            group_balance=group_balance,
            retention_coef=retention_coef,
            retention_margin=retention_margin,
            retention_role=retention_role,
            retention_reference_top_only=retention_reference_top_only,
            top_action_retention_coef=top_action_retention_coef,
            top_action_retention_margin=top_action_retention_margin,
            top_action_retention_role=top_action_retention_role,
            top_action_retention_reference_top_only=top_action_retention_reference_top_only,
        )


__all__ = [
    "ImpalaUpdateLoopMixin",
    "ScopedOptimizerUpdateSpec",
    "run_scoped_impala_optimizer_update",
]
