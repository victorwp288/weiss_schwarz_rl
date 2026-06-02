"""IMPALA post-forward loss-core orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_context_stage import finalize_impala_loss_context_stage
from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs
from weiss_rl.learners.impala.loss_metrics_stage import assemble_impala_loss_core_metrics
from weiss_rl.learners.impala.loss_objective_stage import (
    compute_impala_objective_stage,
    resolve_impala_value_loss_mask,
)
from weiss_rl.learners.impala.loss_policy_anchor_stage import apply_impala_policy_anchor_stage
from weiss_rl.learners.impala.loss_teacher_stage import apply_impala_teacher_auxiliary_stage
from weiss_rl.learners.impala.loss_vtrace_stage import (
    ImpalaVTraceClipConfig,
    attach_resolved_vtrace_context,
    compute_impala_vtrace_stage,
    resolve_impala_vtrace_clip_config,
)

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaLossCoreResult:
    total_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Any]


def compute_impala_loss_core(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    action_logp: Tensor,
    entropy: Tensor,
    batch_value: BatchValueGetter,
) -> ImpalaLossCoreResult:
    vtrace_stage = compute_impala_vtrace_stage(
        learner=learner,
        batch=batch,
        inputs=inputs,
        action_logp=action_logp,
        batch_value=batch_value,
    )
    action_logp = vtrace_stage.action_logp
    resolved_vtrace = vtrace_stage.resolved_vtrace
    clip_config = vtrace_stage.clip_config
    context = inputs.context

    objective_stage = compute_impala_objective_stage(
        learner=learner,
        batch=batch,
        inputs=inputs,
        policy_action_logp=action_logp,
        retention_action_logp=vtrace_stage.retention_action_logp,
        entropy=entropy,
        resolved_vtrace=resolved_vtrace,
        batch_value=batch_value,
    )
    objective_losses = objective_stage.losses
    total_loss = objective_losses.total_loss

    policy_anchor_stage = apply_impala_policy_anchor_stage(
        learner=learner,
        batch=batch,
        inputs=inputs,
        total_loss=total_loss,
    )
    total_loss = policy_anchor_stage.total_loss

    action_catalog = getattr(learner.model, "action_catalog", None)
    teacher_finalization = apply_impala_teacher_auxiliary_stage(
        learner=learner,
        batch=batch,
        inputs=inputs,
        action_catalog=action_catalog,
        total_loss=total_loss,
        batch_value=batch_value,
    )
    total_loss = teacher_finalization.total_loss

    context = finalize_impala_loss_context_stage(
        learner=learner,
        batch=batch,
        inputs=inputs,
        total_loss=total_loss,
        objective_losses=objective_losses,
        policy_anchor_stage=policy_anchor_stage,
    )

    metrics = assemble_impala_loss_core_metrics(
        learner=learner,
        batch=batch,
        inputs=inputs,
        total_loss=total_loss,
        objective_losses=objective_losses,
        policy_anchor_stage=policy_anchor_stage,
        teacher_finalization=teacher_finalization,
        resolved_vtrace=resolved_vtrace,
        clip_config=clip_config,
        action_logp=action_logp,
        action_catalog=action_catalog,
        batch_value=batch_value,
    )
    return ImpalaLossCoreResult(total_loss=total_loss, metrics=metrics, context=context)


__all__ = [
    "ImpalaLossCoreResult",
    "ImpalaVTraceClipConfig",
    "attach_resolved_vtrace_context",
    "compute_impala_loss_core",
    "resolve_impala_value_loss_mask",
    "resolve_impala_vtrace_clip_config",
]
