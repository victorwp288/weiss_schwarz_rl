from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher.packed import compute_packed_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.packed_margins import compute_packed_teacher_margin_supervision

from .teacher_public_margin_test_support import _packed_teacher_public_margin_case


def test_compute_packed_teacher_margin_supervision_matches_packed_branch_margin_terms() -> None:
    case = _packed_teacher_public_margin_case(logits=(0.0, 2.0, -1.0), teacher_action_id=5)

    direct = compute_packed_teacher_margin_supervision(
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
        action_margin_coef=0.25,
        action_margin=0.5,
        same_family_action_margin_coef=0.75,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=case.zero,
        value_dtype=case.packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_margin_loss = direct.action_margin_loss * 0.25 + direct.same_family_action_margin_loss * 0.75

    torch.testing.assert_close(packed_loss, expected_margin_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)
