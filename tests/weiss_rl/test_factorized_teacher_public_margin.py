from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.packed_margins import compute_packed_teacher_margin_supervision
from weiss_rl.learners.structured_teacher.packed_public import compute_packed_teacher_public_supervision

from .teacher_public_margin_test_support import _packed_teacher_public_margin_case


def test_factorized_structured_teacher_reuses_packed_public_and_margin_helpers() -> None:
    case = _packed_teacher_public_margin_case(logits=(0.0, 2.0, -1.0), teacher_action_id=0)
    main_play_family_id = case.family_index["main_play_character"]
    family_logits = torch.full((1, 1, len(case.action_catalog.families)), -3.0, dtype=torch.float32)
    family_logits[0, 0, main_play_family_id] = 3.0
    margin_direct = compute_packed_teacher_margin_supervision(
        packed_view=case.packed_view,
        flat_teacher_action=case.teacher_action.reshape(-1),
        flat_teacher_family=case.teacher_family.reshape(-1),
        flat_teacher_valid=case.teacher_valid.reshape(-1),
        flat_loss_mask=case.loss_mask.reshape(-1),
        exact_action_family_rows=None,
        action_margin_coef=1.0,
        action_margin=0.5,
        same_family_action_margin_coef=1.0,
        same_family_action_margin=0.5,
        zero=case.zero,
        value_dtype=case.packed_view.logits.dtype,
    )
    public_direct = compute_packed_teacher_public_supervision(
        packed_view=case.packed_view,
        public_heuristic_target_logits=case.public_target_logits,
        public_heuristic_family_ids=(main_play_family_id,),
        flat_teacher_family=case.teacher_family.reshape(-1),
        flat_teacher_valid=case.teacher_valid.reshape(-1),
        flat_loss_mask=case.loss_mask.reshape(-1),
        pass_action_id=case.action_catalog.pass_action_id,
        public_heuristic_coef=1.0,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=1.0,
        public_nonpass_over_pass_margin=0.5,
        zero=case.zero,
        value_dtype=case.packed_view.logits.dtype,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=case.teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=case.teacher_action,
        teacher_valid=case.teacher_valid,
        loss_mask=case.loss_mask,
        action_catalog=case.action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.2,
        action_margin=0.5,
        same_family_action_margin_coef=0.4,
        same_family_action_margin=0.5,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=case.public_target_logits,
        packed_view=case.packed_view,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )
    expected_loss = (
        margin_direct.action_margin_loss * 0.2
        + margin_direct.same_family_action_margin_loss * 0.4
        + public_direct.public_heuristic_loss * 0.7
        + public_direct.public_nonpass_over_pass_loss * 0.3
    )

    torch.testing.assert_close(aux_loss, expected_loss)
    expected_metrics = {**margin_direct.metrics, **public_direct.metrics}
    for key, value in expected_metrics.items():
        assert metrics[key] == pytest.approx(value)
    expected_context = {**margin_direct.context, **public_direct.context}
    for key, value in expected_context.items():
        torch.testing.assert_close(context[key], value)
