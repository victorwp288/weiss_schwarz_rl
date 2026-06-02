"""Structured-teacher auxiliary request assembly for IMPALA learners."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala.teacher_auxiliary_call import (
    compute_structured_teacher_auxiliary_from_impala_inputs,
)
from weiss_rl.learners.impala.teacher_auxiliary_inputs import (
    BatchValueGetter,
    ImpalaTeacherAuxiliaryCoefficients,
    ImpalaTeacherAuxiliaryFactorizedInputs,
    ImpalaTeacherAuxiliaryInputs,
    ImpalaTeacherAuxiliaryLabels,
    ImpalaTeacherAuxiliaryPackedInputs,
    resolve_impala_teacher_auxiliary_coefficients,
    resolve_impala_teacher_auxiliary_factorized_inputs,
    resolve_impala_teacher_auxiliary_inputs,
    resolve_impala_teacher_auxiliary_labels,
    resolve_impala_teacher_auxiliary_packed_inputs,
)


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryResult:
    loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Any]


def compute_impala_teacher_auxiliary(
    *,
    learner: Any,
    batch: Any,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    expected_shape: torch.Size,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    packed_view: Any,
    factorized_result: Any,
    public_heuristic_target_logits: Tensor | None,
    batch_value: BatchValueGetter,
) -> ImpalaTeacherAuxiliaryResult:
    inputs = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
        expected_shape=expected_shape,
        packed_legal=packed_legal,
        packed_view=packed_view,
        factorized_result=factorized_result,
    )

    teacher_aux_started = time.perf_counter()
    teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_from_impala_inputs(
        inputs=inputs,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        public_heuristic_target_logits=public_heuristic_target_logits,
    )
    learner._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
    return ImpalaTeacherAuxiliaryResult(
        loss=teacher_aux_loss,
        metrics=teacher_metrics,
        context=teacher_context,
    )


__all__ = [
    "ImpalaTeacherAuxiliaryCoefficients",
    "ImpalaTeacherAuxiliaryFactorizedInputs",
    "ImpalaTeacherAuxiliaryInputs",
    "ImpalaTeacherAuxiliaryLabels",
    "ImpalaTeacherAuxiliaryPackedInputs",
    "ImpalaTeacherAuxiliaryResult",
    "compute_impala_teacher_auxiliary",
    "compute_structured_teacher_auxiliary_from_impala_inputs",
    "resolve_impala_teacher_auxiliary_coefficients",
    "resolve_impala_teacher_auxiliary_factorized_inputs",
    "resolve_impala_teacher_auxiliary_inputs",
    "resolve_impala_teacher_auxiliary_labels",
    "resolve_impala_teacher_auxiliary_packed_inputs",
]
