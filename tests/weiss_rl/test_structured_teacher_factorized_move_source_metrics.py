from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)

from .impala_test_support import _teacher_aux_catalog
from .structured_teacher_factorized_metrics_test_support import confident_family_log_probs, family_indices


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_move_source_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = family_indices(action_catalog)
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=confident_family_log_probs(action_catalog, ["main_move"]),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_explicit_move_source_labels() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = family_indices(action_catalog)
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[-1]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=confident_family_log_probs(action_catalog, ["main_move"]),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
        teacher_move_source=torch.tensor([[int(move_decoded.from_slot or 0)]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0
