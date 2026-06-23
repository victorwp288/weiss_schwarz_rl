from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)

from .impala_test_support import _teacher_aux_catalog
from .structured_teacher_factorized_metrics_test_support import family_indices


def test_compute_structured_teacher_auxiliary_metrics_reports_family_coverage_on_active_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = family_indices(action_catalog)
    family_logits = torch.zeros((4, 1, len(action_catalog.families)), dtype=torch.float32)
    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [
                [family_index["main_play_character"]],
                [family_index["main_move"]],
                [family_index["attack"]],
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [1], [0], [2]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1], [0], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [5], [11], [5]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True], [True], [False]], dtype=torch.bool),
        loss_mask=torch.tensor([[1.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )

    assert metrics["teacher_active_fraction"] == pytest.approx(0.75)
    assert metrics["teacher_main_play_character_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_main_move_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_attack_fraction"] == pytest.approx(0.0)
