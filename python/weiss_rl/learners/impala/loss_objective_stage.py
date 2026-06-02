"""IMPALA objective-loss stage orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs
from weiss_rl.learners.impala.objective_loss import ImpalaObjectiveLosses, compute_impala_objective_losses

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaObjectiveStage:
    losses: ImpalaObjectiveLosses


def resolve_impala_value_loss_mask(
    *,
    learner: Any,
    batch: Any,
    expected_shape: Any,
    like: Tensor,
    batch_value: BatchValueGetter,
) -> Tensor | None:
    return learner._optional_time_major_loss_mask(
        batch_value(batch, "value_train_mask"),
        expected_shape=expected_shape,
        like=like,
    )


def compute_impala_objective_stage(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    policy_action_logp: Tensor,
    retention_action_logp: Tensor,
    entropy: Tensor,
    resolved_vtrace: Any,
    batch_value: BatchValueGetter,
) -> ImpalaObjectiveStage:
    value_loss_mask = resolve_impala_value_loss_mask(
        learner=learner,
        batch=batch,
        expected_shape=inputs.obs.shape[:2],
        like=inputs.values,
        batch_value=batch_value,
    )
    objective_losses = compute_impala_objective_losses(
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        actions=inputs.actions,
        advantages=resolved_vtrace.advantages,
        values=inputs.values,
        targets=resolved_vtrace.targets,
        entropy=entropy,
        loss_mask=inputs.loss_mask,
        value_loss_mask=value_loss_mask,
        value_loss_coef=float(learner.value_loss_coef),
        entropy_coef=float(learner.entropy_coef),
        trajectory_retention_valid=inputs.trajectory_retention_valid,
        trajectory_retention_coef=float(learner.trajectory_retention_coef),
        top_action_ids=None
        if inputs.factorized_result is None
        else getattr(inputs.factorized_result, "top_action_ids", None),
    )
    if float(learner.trajectory_retention_coef) != 0.0:
        inputs.context["trajectory_retention_loss"] = objective_losses.trajectory_retention_loss.detach()
    inputs.context["value_train_mask"] = objective_losses.value_loss_mask.detach()
    return ImpalaObjectiveStage(losses=objective_losses)


__all__ = ["ImpalaObjectiveStage", "compute_impala_objective_stage", "resolve_impala_value_loss_mask"]
