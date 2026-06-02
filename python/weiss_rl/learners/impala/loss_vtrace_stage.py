"""IMPALA V-trace loss-stage orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs
from weiss_rl.learners.impala.vtrace_targets import resolve_impala_vtrace_targets

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaVTraceClipConfig:
    rho_bar: float
    c_bar: float


@dataclass(frozen=True, slots=True)
class ImpalaVTraceStage:
    retention_action_logp: Tensor
    action_logp: Tensor
    clip_config: ImpalaVTraceClipConfig
    resolved_vtrace: Any


def resolve_impala_vtrace_clip_config(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
) -> ImpalaVTraceClipConfig:
    rho_bar_value = batch_value(batch, "vtrace_rho_bar")
    c_bar_value = batch_value(batch, "vtrace_c_bar")
    return ImpalaVTraceClipConfig(
        rho_bar=learner.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
        c_bar=learner.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
    )


def attach_resolved_vtrace_context(
    *,
    context: dict[str, Any],
    resolved_vtrace: Any,
    loss_mask: Tensor,
) -> None:
    context["targets"] = resolved_vtrace.targets.detach()
    context["advantages"] = resolved_vtrace.advantages.detach()
    context["vtrace_rhos"] = resolved_vtrace.rhos_for_metrics.detach()
    context["rewards"] = resolved_vtrace.rewards_for_metrics.detach()
    context["policy_train_mask"] = loss_mask.detach()


def compute_impala_vtrace_stage(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    action_logp: Tensor,
    batch_value: BatchValueGetter,
) -> ImpalaVTraceStage:
    retention_action_logp = action_logp
    clip_config = resolve_impala_vtrace_clip_config(learner=learner, batch=batch, batch_value=batch_value)
    resolved_vtrace = resolve_impala_vtrace_targets(
        batch=batch,
        vtrace_result=inputs.vtrace_result,
        values=inputs.values,
        action_logp=action_logp,
        loss_mask=inputs.loss_mask,
        rho_bar=clip_config.rho_bar,
        c_bar=clip_config.c_bar,
        float_target=learner._float_target,
        resolve_bootstrap_value=learner._resolve_vtrace_bootstrap_value,
        batch_value=batch_value,
    )
    attach_resolved_vtrace_context(
        context=inputs.context,
        resolved_vtrace=resolved_vtrace,
        loss_mask=inputs.loss_mask,
    )
    return ImpalaVTraceStage(
        retention_action_logp=retention_action_logp,
        action_logp=resolved_vtrace.action_logp,
        clip_config=clip_config,
        resolved_vtrace=resolved_vtrace,
    )


__all__ = [
    "ImpalaVTraceClipConfig",
    "ImpalaVTraceStage",
    "attach_resolved_vtrace_context",
    "compute_impala_vtrace_stage",
    "resolve_impala_vtrace_clip_config",
]
