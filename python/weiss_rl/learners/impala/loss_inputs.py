"""IMPALA loss-input preparation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from weiss_rl.learners.impala.loss_assembly import ImpalaLossInputs, assemble_impala_loss_inputs
from weiss_rl.learners.impala.loss_batch_inputs import resolve_impala_loss_batch_inputs
from weiss_rl.learners.impala.loss_forward_context import build_impala_forward_context
from weiss_rl.learners.impala.loss_legal_mask import resolve_impala_dense_legal_mask
from weiss_rl.learners.impala.loss_masks import (
    ImpalaLossForwardFlags,
    ImpalaLossMasks,
    resolve_impala_loss_forward_flags,
    resolve_impala_loss_masks,
)
from weiss_rl.learners.impala.loss_policy_forward import (
    ImpalaPolicyForwardResult,
    evaluate_impala_policy_forward,
)
from weiss_rl.learners.impala.loss_teacher_targets_stage import prepare_impala_loss_teacher_target_inputs

BatchValueGetter = Callable[[Any, str], Any]


def prepare_impala_loss_inputs(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
) -> ImpalaLossInputs:
    batch_inputs = resolve_impala_loss_batch_inputs(learner=learner, batch=batch, batch_value=batch_value)
    masks = resolve_impala_loss_masks(
        learner=learner,
        batch=batch,
        obs=batch_inputs.obs,
        batch_value=batch_value,
    )
    forward_flags = resolve_impala_loss_forward_flags(
        learner=learner,
        packed_legal=batch_inputs.packed_legal,
        loss_mask=masks.loss_mask,
    )
    forward_result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
        forward_model=batch_inputs.forward_model,
        obs=batch_inputs.obs,
        actions=batch_inputs.actions,
        packed_legal=batch_inputs.packed_legal,
        loss_mask=masks.loss_mask,
        reset_before_step=masks.reset_before_step,
        trajectory_retention_active=masks.trajectory_retention_active,
        restrict_packed_policy_rows=forward_flags.restrict_packed_policy_rows,
    )
    legal_mask = resolve_impala_dense_legal_mask(
        learner=learner,
        batch=batch,
        obs=batch_inputs.obs,
        packed_legal=forward_result.packed_legal,
        logits=forward_result.logits,
    )
    teacher_target_inputs = prepare_impala_loss_teacher_target_inputs(
        learner=learner,
        batch=batch,
        forward_model=batch_inputs.forward_model,
        obs=batch_inputs.obs,
        masks=masks,
        forward_flags=forward_flags,
        forward_result=forward_result,
    )
    context = build_impala_forward_context(learner=learner, batch=batch, forward_result=forward_result)

    return assemble_impala_loss_inputs(
        batch_inputs=batch_inputs,
        masks=masks,
        forward_flags=forward_flags,
        forward_result=forward_result,
        legal_mask=legal_mask,
        teacher_target_inputs=teacher_target_inputs,
        context=context,
    )


__all__ = [
    "ImpalaLossForwardFlags",
    "ImpalaLossInputs",
    "ImpalaLossMasks",
    "ImpalaPolicyForwardResult",
    "assemble_impala_loss_inputs",
    "evaluate_impala_policy_forward",
    "prepare_impala_loss_inputs",
    "resolve_impala_loss_forward_flags",
    "resolve_impala_loss_masks",
]
