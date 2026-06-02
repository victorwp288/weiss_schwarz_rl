"""Structured-teacher auxiliary optimizer-update orchestration."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.scoped_optimizer_update import (
    ScopedOptimizerUpdateSpec,
    run_scoped_impala_optimizer_update,
)


def run_impala_auxiliary_optimizer_update(*, learner: Any, batch: Any) -> dict[str, float]:
    return run_scoped_impala_optimizer_update(
        learner=learner,
        batch=batch,
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message="ImpalaLearner requires a model to run an auxiliary optimizer step",
            loss_timer_name="learner_auxiliary_loss_and_metrics",
        ),
        build_loss=lambda: learner._auxiliary_loss_and_metrics(batch),
    )


__all__ = ["run_impala_auxiliary_optimizer_update"]
