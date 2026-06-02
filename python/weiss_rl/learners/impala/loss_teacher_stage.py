"""IMPALA teacher-auxiliary loss-stage orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.loss_finalization import (
    ImpalaLossFinalization,
    apply_impala_teacher_auxiliary,
)
from weiss_rl.learners.impala.loss_inputs import ImpalaLossInputs

BatchValueGetter = Callable[[Any, str], Any]


def apply_impala_teacher_auxiliary_stage(
    *,
    learner: Any,
    batch: Any,
    inputs: ImpalaLossInputs,
    total_loss: Tensor,
    action_catalog: Any,
    batch_value: BatchValueGetter,
) -> ImpalaLossFinalization:
    return apply_impala_teacher_auxiliary(
        learner=learner,
        batch=batch,
        total_loss=total_loss,
        context=inputs.context,
        teacher_aux_active=inputs.teacher_aux_active,
        logits=inputs.logits,
        legal_mask=inputs.legal_mask,
        loss_mask=inputs.loss_mask,
        action_catalog=action_catalog,
        expected_shape=inputs.values.shape,
        packed_legal=inputs.packed_legal,
        packed_view=inputs.teacher_aux_packed_view,
        factorized_result=inputs.factorized_result,
        public_heuristic_target_logits=inputs.public_heuristic_target_logits,
        resolve_legal_mask=lambda source_batch, expected_shape, action_dim: learner._resolve_legal_mask(
            source_batch,
            expected_shape=expected_shape,
            action_dim=action_dim,
        ),
        batch_value=batch_value,
    )


__all__ = ["apply_impala_teacher_auxiliary_stage"]
