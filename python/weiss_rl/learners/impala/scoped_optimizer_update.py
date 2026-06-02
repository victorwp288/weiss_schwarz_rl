"""Shared scoped optimizer-update orchestration for IMPALA learners."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from weiss_rl.learners.impala.optimizer_step import run_impala_optimizer_step
from weiss_rl.learners.impala.update_bookkeeping import (
    begin_impala_update_scope,
    finalize_impala_update_scope,
)
from weiss_rl.learners.impala.update_loss_stage import ScopedLossBuilder, build_scoped_impala_loss


@dataclass(frozen=True, slots=True)
class ScopedOptimizerUpdateSpec:
    missing_model_message: str
    loss_timer_name: str
    count_learner_update: bool = False
    include_training_metrics: bool = False
    checkpoint_on_interval: bool = False
    scale_loss_on_nonfinite_gradients: bool = False


def run_scoped_impala_optimizer_update(
    *,
    learner: Any,
    batch: Any,
    spec: ScopedOptimizerUpdateSpec,
    build_loss: ScopedLossBuilder,
    started_at: float | None = None,
) -> dict[str, float]:
    if learner.model is None:
        raise ValueError(spec.missing_model_message)
    update_started = time.perf_counter() if started_at is None else float(started_at)
    update_scope = begin_impala_update_scope(
        learner=learner,
        batch=batch,
        started_at=update_started,
        count_learner_update=spec.count_learner_update,
        include_training_metrics=spec.include_training_metrics,
        checkpoint_on_interval=spec.checkpoint_on_interval,
    )
    loss_build = build_scoped_impala_loss(
        learner=learner,
        loss_timer_name=spec.loss_timer_name,
        build_loss=build_loss,
    )
    metrics = run_impala_optimizer_step(
        learner=learner,
        batch=batch,
        loss=loss_build.loss,
        base_metrics=loss_build.metrics,
        context=loss_build.context,
        scale_loss_on_nonfinite_gradients=spec.scale_loss_on_nonfinite_gradients,
    )
    return finalize_impala_update_scope(
        learner=learner,
        metrics=metrics,
        started_at=update_scope.started_at,
    )


__all__ = ["ScopedOptimizerUpdateSpec", "run_scoped_impala_optimizer_update"]
