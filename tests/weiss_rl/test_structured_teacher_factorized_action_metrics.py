from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)

from .impala_test_support import _teacher_aux_catalog
from .structured_teacher_factorized_metrics_test_support import confident_family_log_probs, family_indices


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_same_family_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = family_indices(action_catalog)
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["main_move"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
        factorized_family_log_probs=confident_family_log_probs(
            action_catalog,
            ["main_play_character", "main_move"],
        ),
        factorized_same_family_action_logp=torch.tensor([[-0.1], [-0.2]], dtype=torch.float32),
        factorized_same_family_top_action_ids=torch.tensor([[0], [move_action]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(1.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_exact_action_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = family_indices(action_catalog)
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        factorized_family_log_probs=confident_family_log_probs(action_catalog, ["main_play_character"]),
        factorized_top_action_ids=torch.tensor([[0]], dtype=torch.long),
        factorized_same_family_action_logp=torch.tensor([[-0.1]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_loss"] > 0.0
