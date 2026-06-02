"""IMPALA loss metrics-stage request assembly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_finalization import ImpalaLossFinalization
from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs
from weiss_rl.learners.impala.loss_policy_anchor_stage import ImpalaPolicyAnchorStage
from weiss_rl.learners.impala.metrics_assembly import (
    ImpalaMetricAssemblyRequest,
    assemble_impala_loss_metrics,
)
from weiss_rl.learners.impala.objective_loss import ImpalaObjectiveLosses

BatchValueGetter = Callable[[Any, str], Any]


def assemble_impala_loss_core_metrics(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    total_loss: Tensor,
    objective_losses: ImpalaObjectiveLosses,
    policy_anchor_stage: ImpalaPolicyAnchorStage,
    teacher_finalization: ImpalaLossFinalization,
    resolved_vtrace: Any,
    clip_config: Any,
    action_logp: Tensor,
    action_catalog: Any,
    batch_value: BatchValueGetter,
) -> dict[str, float]:
    return assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=total_loss,
            policy_loss=objective_losses.policy_loss,
            value_loss=objective_losses.value_loss,
            entropy_mean=objective_losses.entropy_mean,
            entropy_scope=learner.entropy_scope,
            loss_mask=inputs.loss_mask,
            value_loss_mask=objective_losses.value_loss_mask,
            actions=inputs.actions,
            action_logp=action_logp,
            behavior_logp_for_mask=resolved_vtrace.behavior_logp_for_mask,
            rewards_for_metrics=resolved_vtrace.rewards_for_metrics,
            advantages=resolved_vtrace.advantages,
            targets=resolved_vtrace.targets,
            rhos_for_metrics=resolved_vtrace.rhos_for_metrics,
            rho_bar=clip_config.rho_bar,
            c_bar=clip_config.c_bar,
            action_catalog=action_catalog,
            pass_action_id=learner.pass_action_id,
            trajectory_retention_metrics=objective_losses.trajectory_retention_metrics,
            policy_anchor_metrics=policy_anchor_stage.policy_anchor_metrics,
            teacher_metrics=teacher_finalization.teacher_metrics,
            emit_structured_metrics=inputs.emit_structured_metrics,
            logits=inputs.logits,
            legal_mask=inputs.legal_mask,
            packed_legal=inputs.packed_legal,
            packed_view=inputs.packed_view,
            factorized_result=inputs.factorized_result,
            batch=batch,
            expected_shape=inputs.obs.shape[:2],
            action_dim=None if inputs.logits is None else int(inputs.logits.shape[-1]),
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: learner._resolve_legal_mask(
                source_batch,
                expected_shape=expected_shape,
                action_dim=action_dim,
            ),
        ),
        batch_value=batch_value,
        record_timing_ms=learner._record_timing_ms,
    )


__all__ = ["assemble_impala_loss_core_metrics"]
