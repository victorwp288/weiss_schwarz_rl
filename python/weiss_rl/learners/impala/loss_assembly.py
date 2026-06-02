"""Final IMPALA loss-input assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_batch_inputs import ImpalaLossBatchInputs
from weiss_rl.learners.impala.loss_masks import ImpalaLossForwardFlags, ImpalaLossMasks
from weiss_rl.learners.impala.loss_policy_forward import ImpalaPolicyForwardResult
from weiss_rl.learners.impala.teacher_target_inputs import ImpalaTeacherTargetInputs
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView


@dataclass(frozen=True, slots=True)
class ImpalaLossInputs:
    vtrace_result: Any
    obs: Tensor
    actions: Tensor
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None
    forward_model: Any
    loss_mask: Tensor
    reset_before_step: Tensor | None
    trajectory_retention_valid: Tensor | None
    teacher_aux_active: bool
    emit_structured_metrics: bool
    factorized_result: Any
    logits: Tensor | None
    packed_logits: Tensor | None
    values: Tensor
    forward_observation_context: Mapping[str, Tensor] | None
    legal_mask: Tensor | None
    packed_view: PackedStructuredLegalView | None
    teacher_aux_packed_view: PackedStructuredLegalView | None
    public_heuristic_target_logits: Tensor | None
    context: dict[str, Any]


def assemble_impala_loss_inputs(
    *,
    batch_inputs: ImpalaLossBatchInputs,
    masks: ImpalaLossMasks,
    forward_flags: ImpalaLossForwardFlags,
    forward_result: ImpalaPolicyForwardResult,
    legal_mask: Tensor | None,
    teacher_target_inputs: ImpalaTeacherTargetInputs,
    context: dict[str, Any],
) -> ImpalaLossInputs:
    return ImpalaLossInputs(
        vtrace_result=batch_inputs.vtrace_result,
        obs=batch_inputs.obs,
        actions=batch_inputs.actions,
        packed_legal=forward_result.packed_legal,
        forward_model=batch_inputs.forward_model,
        loss_mask=masks.loss_mask,
        reset_before_step=masks.reset_before_step,
        trajectory_retention_valid=masks.trajectory_retention_valid,
        teacher_aux_active=forward_flags.teacher_aux_active,
        emit_structured_metrics=forward_flags.emit_structured_metrics,
        factorized_result=forward_result.factorized_result,
        logits=forward_result.logits,
        packed_logits=forward_result.packed_logits,
        values=forward_result.values,
        forward_observation_context=forward_result.forward_observation_context,
        legal_mask=legal_mask,
        packed_view=teacher_target_inputs.packed_view,
        teacher_aux_packed_view=teacher_target_inputs.teacher_aux_packed_view,
        public_heuristic_target_logits=teacher_target_inputs.public_heuristic_target_logits,
        context=context,
    )


__all__ = ["ImpalaLossInputs", "assemble_impala_loss_inputs"]
