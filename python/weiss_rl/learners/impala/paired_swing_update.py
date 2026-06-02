"""Paired-swing replay optimizer-update orchestration."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.scoped_optimizer_update import (
    ScopedOptimizerUpdateSpec,
    run_scoped_impala_optimizer_update,
)


def run_impala_paired_swing_optimizer_update(
    *,
    learner: Any,
    batch: Any,
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
    if float(full_surface_top_action_retention_coef) < 0.0:
        raise ValueError("full_surface_top_action_retention_coef must be >= 0")
    if float(full_surface_top_action_retention_margin) < 0.0:
        raise ValueError("full_surface_top_action_retention_margin must be >= 0")
    if float(full_surface_top_action_retention_coef) != 0.0 and full_surface_retention_batch is None:
        raise ValueError("full_surface_retention_batch is required when full-surface retention is active")

    def build_loss() -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        loss, swing_metrics, swing_context = learner._paired_swing_loss_and_metrics(
            batch,
            margin=float(margin),
            coef=float(coef),
            positive_action_source=str(positive_action_source),
            negative_action_source=str(negative_action_source),
            loss_scope=str(loss_scope),
            compare_to=str(compare_to),
            margin_retention_coef=float(margin_retention_coef),
            margin_retention_margin=float(margin_retention_margin),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
        )
        if full_surface_retention_batch is not None and float(full_surface_top_action_retention_coef) != 0.0:
            retention_loss, retention_metrics, retention_context = (
                learner._paired_swing_full_surface_top_action_retention_loss_and_metrics(
                    full_surface_retention_batch,
                    coef=float(full_surface_top_action_retention_coef),
                    margin=float(full_surface_top_action_retention_margin),
                    mode=str(full_surface_top_action_retention_mode),
                )
            )
            loss = loss + retention_loss
            swing_metrics.update(retention_metrics)
            swing_context.update(retention_context)
        return loss, swing_metrics, swing_context

    return run_scoped_impala_optimizer_update(
        learner=learner,
        batch=batch,
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message="ImpalaLearner requires a model to run a paired-swing optimizer step",
            loss_timer_name="learner_paired_swing_loss_and_metrics",
        ),
        build_loss=build_loss,
    )


__all__ = ["run_impala_paired_swing_optimizer_update"]
