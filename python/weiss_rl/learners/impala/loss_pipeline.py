"""End-to-end IMPALA loss pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.action_reductions import resolve_impala_action_reductions
from weiss_rl.learners.impala.loss_core import compute_impala_loss_core
from weiss_rl.learners.impala.loss_inputs import prepare_impala_loss_inputs

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaLossActionReductionResult:
    action_logp: Tensor
    entropy: Tensor
    context: dict[str, Any]


def resolve_impala_loss_action_reductions(
    *,
    learner: Any,
    batch: Any,
    loss_inputs: Any,
) -> ImpalaLossActionReductionResult:
    action_reductions = resolve_impala_action_reductions(
        factorized_result=loss_inputs.factorized_result,
        logits=loss_inputs.logits,
        packed_logits=loss_inputs.packed_logits,
        legal_mask=loss_inputs.legal_mask,
        packed_legal=loss_inputs.packed_legal,
        actions=loss_inputs.actions,
        entropy_scope=learner.entropy_scope,
        pass_action_id=learner.pass_action_id,
        action_catalog=getattr(learner.model, "action_catalog", None),
        record_timing_ms=learner._record_timing_ms,
    )
    action_logp = action_reductions.action_logp
    entropy = action_reductions.entropy
    context = loss_inputs.context
    context["action_logp"] = action_logp.detach()
    context["entropy"] = entropy.detach()
    learner._ensure_finite_tensor("action_logp", action_logp, batch=batch, context=context)
    learner._ensure_finite_tensor("entropy", entropy, batch=batch, context=context)
    return ImpalaLossActionReductionResult(action_logp=action_logp, entropy=entropy, context=context)


def compute_impala_loss_and_metrics_with_context(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
    loss_inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=batch_value)
    action_reductions = resolve_impala_loss_action_reductions(
        learner=learner,
        batch=batch,
        loss_inputs=loss_inputs,
    )
    loss_core = compute_impala_loss_core(
        learner=learner,
        batch=batch,
        inputs=loss_inputs,
        action_logp=action_reductions.action_logp,
        entropy=action_reductions.entropy,
        batch_value=batch_value,
    )
    return loss_core.total_loss, loss_core.metrics, loss_core.context


__all__ = [
    "ImpalaLossActionReductionResult",
    "compute_impala_loss_and_metrics_with_context",
    "resolve_impala_loss_action_reductions",
]
