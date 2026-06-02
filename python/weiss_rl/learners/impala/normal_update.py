"""Normal IMPALA learner-update orchestration."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.update_bookkeeping import (
    begin_impala_update_scope,
    finalize_impala_update_scope,
)
from weiss_rl.learners.impala.update_logging import log_impala_update_metrics_if_due
from weiss_rl.learners.impala.update_training_inputs import (
    has_impala_training_inputs,
    resolve_impala_update_vtrace_result,
    summarize_precomputed_vtrace_update_metrics,
)
from weiss_rl.learners.impala.update_training_step import run_impala_training_optimizer_step


def run_impala_normal_update(*, learner: Any, batch: Any) -> dict[str, float]:
    update_scope = begin_impala_update_scope(
        learner=learner,
        batch=batch,
        count_learner_update=True,
        include_training_metrics=True,
        checkpoint_on_interval=True,
    )
    metrics = update_scope.metrics
    vtrace_result = resolve_impala_update_vtrace_result(batch)

    if has_impala_training_inputs(batch):
        metrics.update(run_impala_training_optimizer_step(learner=learner, batch=batch))

    metrics.update(
        summarize_precomputed_vtrace_update_metrics(
            learner=learner,
            batch=batch,
            vtrace_result=vtrace_result,
        )
    )

    log_impala_update_metrics_if_due(learner=learner, batch=batch, metrics=metrics)

    return finalize_impala_update_scope(
        learner=learner,
        metrics=metrics,
        started_at=update_scope.started_at,
    )


__all__ = ["run_impala_normal_update"]
