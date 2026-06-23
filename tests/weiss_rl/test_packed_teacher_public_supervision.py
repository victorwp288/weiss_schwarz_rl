from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher.packed import compute_packed_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.packed_public import compute_packed_teacher_public_supervision

from .teacher_public_margin_test_support import _packed_teacher_public_margin_case


def test_compute_packed_teacher_public_supervision_matches_packed_branch_public_terms() -> None:
    case = _packed_teacher_public_margin_case(logits=(0.0, -0.5, 3.0), teacher_action_id=0)
    main_play_family_id = case.family_index["main_play_character"]

    direct = compute_packed_teacher_public_supervision(
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
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=case.packed_view,
        packed_offsets=case.packed_offsets,
        teacher_family=case.teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=case.teacher_action,
        teacher_valid=case.teacher_valid,
        teacher_move_source=None,
        loss_mask=case.loss_mask,
        action_catalog=case.action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=case.public_target_logits,
        zero=case.zero,
        value_dtype=case.packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_public_loss = direct.public_heuristic_loss * 0.7 + direct.public_nonpass_over_pass_loss * 0.3

    torch.testing.assert_close(packed_loss, expected_public_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)
