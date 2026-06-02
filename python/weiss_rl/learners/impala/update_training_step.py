"""Normal IMPALA training optimizer-step assembly."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.optimizer_step import run_impala_optimizer_step
from weiss_rl.learners.impala.update_loss_stage import build_scoped_impala_loss
from weiss_rl.learners.impala.update_training_inputs import validate_impala_training_inputs


def run_impala_training_optimizer_step(*, learner: Any, batch: Any) -> dict[str, float]:
    validate_impala_training_inputs(learner=learner, batch=batch)
    if learner.model is None:
        raise ValueError("ImpalaLearner requires a model to run an optimizer step")

    loss_build = build_scoped_impala_loss(
        learner=learner,
        loss_timer_name="learner_loss_and_metrics",
        build_loss=lambda: learner._loss_and_metrics_with_context(batch),
    )
    return run_impala_optimizer_step(
        learner=learner,
        batch=batch,
        loss=loss_build.loss,
        base_metrics=loss_build.metrics,
        context=loss_build.context,
        scale_loss_on_nonfinite_gradients=True,
    )


__all__ = ["run_impala_training_optimizer_step"]
