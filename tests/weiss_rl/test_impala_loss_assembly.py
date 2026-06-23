from __future__ import annotations

from typing import Any, cast

import torch
from weiss_rl.learners.impala.loss_assembly import assemble_impala_loss_inputs
from weiss_rl.learners.impala.loss_batch_inputs import ImpalaLossBatchInputs
from weiss_rl.learners.impala.loss_masks import ImpalaLossForwardFlags, ImpalaLossMasks
from weiss_rl.learners.impala.loss_policy_forward import ImpalaPolicyForwardResult
from weiss_rl.learners.impala.teacher_target_inputs import ImpalaTeacherTargetInputs


def test_assemble_impala_loss_inputs_preserves_stage_outputs_by_identity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    reset_before_step = torch.zeros((2, 1), dtype=torch.bool)
    trajectory_retention_valid = torch.ones((2, 1), dtype=torch.float32)
    logits = torch.zeros((2, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((3,), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    public_target_logits = torch.full((3,), 0.25, dtype=torch.float32)
    original_packed_legal = (torch.as_tensor([0]), torch.as_tensor([0, 1]), None)
    forward_packed_legal = (torch.as_tensor([1, 2, 3]), torch.as_tensor([0, 3]), None)
    forward_model = object()
    factorized_result = object()
    observation_context = {"obs": obs.reshape(-1, obs.shape[-1])}
    context = {"logits": logits, "values": values}
    packed_view = cast(Any, object())
    teacher_aux_packed_view = cast(Any, object())

    assembled = assemble_impala_loss_inputs(
        batch_inputs=ImpalaLossBatchInputs(
            vtrace_result="vtrace",
            obs=obs,
            actions=actions,
            packed_legal=original_packed_legal,
            forward_model=forward_model,
        ),
        masks=ImpalaLossMasks(
            loss_mask=loss_mask,
            reset_before_step=reset_before_step,
            trajectory_retention_valid=trajectory_retention_valid,
            trajectory_retention_active=None,
        ),
        forward_flags=ImpalaLossForwardFlags(
            teacher_aux_active=True,
            emit_structured_metrics=True,
            restrict_packed_policy_rows=False,
        ),
        forward_result=ImpalaPolicyForwardResult(
            factorized_result=factorized_result,
            packed_legal=forward_packed_legal,
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            forward_observation_context=observation_context,
        ),
        legal_mask=legal_mask,
        teacher_target_inputs=ImpalaTeacherTargetInputs(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_aux_packed_view,
            public_heuristic_target_logits=public_target_logits,
        ),
        context=context,
    )

    assert assembled.vtrace_result == "vtrace"
    assert assembled.obs is obs
    assert assembled.actions is actions
    assert assembled.packed_legal is forward_packed_legal
    assert assembled.packed_legal is not original_packed_legal
    assert assembled.forward_model is forward_model
    assert assembled.loss_mask is loss_mask
    assert assembled.reset_before_step is reset_before_step
    assert assembled.trajectory_retention_valid is trajectory_retention_valid
    assert assembled.teacher_aux_active is True
    assert assembled.emit_structured_metrics is True
    assert assembled.factorized_result is factorized_result
    assert assembled.logits is logits
    assert assembled.packed_logits is packed_logits
    assert assembled.values is values
    assert assembled.forward_observation_context is observation_context
    assert assembled.legal_mask is legal_mask
    assert assembled.packed_view is packed_view
    assert assembled.teacher_aux_packed_view is teacher_aux_packed_view
    assert assembled.public_heuristic_target_logits is public_target_logits
    assert assembled.context is context
