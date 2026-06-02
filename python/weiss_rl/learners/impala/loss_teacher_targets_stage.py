"""IMPALA loss-input teacher-target preparation stage."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.teacher_target_inputs import (
    ImpalaTeacherTargetInputs,
    prepare_impala_teacher_target_inputs,
)


def prepare_impala_loss_teacher_target_inputs(
    *,
    learner: Any,
    batch: Any,
    forward_model: Any,
    obs: Any,
    masks: Any,
    forward_flags: Any,
    forward_result: Any,
) -> ImpalaTeacherTargetInputs:
    return prepare_impala_teacher_target_inputs(
        learner=learner,
        batch=batch,
        forward_model=forward_model,
        obs=obs,
        logits=forward_result.logits,
        packed_logits=forward_result.packed_logits,
        packed_legal=forward_result.packed_legal,
        loss_mask=masks.loss_mask,
        factorized_result=forward_result.factorized_result,
        forward_observation_context=forward_result.forward_observation_context,
        need_packed_view=forward_flags.emit_structured_metrics or forward_flags.teacher_aux_active,
        teacher_aux_enabled=forward_flags.teacher_aux_active,
    )


__all__ = ["prepare_impala_loss_teacher_target_inputs"]
