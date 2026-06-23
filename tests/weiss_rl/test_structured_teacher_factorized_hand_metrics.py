from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import structured_catalog_metadata
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.factorized_hand import compute_factorized_teacher_hand_supervision

from .impala_test_support import _teacher_aux_hand_catalog
from .structured_teacher_factorized_metrics_test_support import (
    confident_family_log_probs,
    family_indices,
    first_action_id,
)


def _hand_actions() -> tuple[object, dict[str, int], int, int, int, int]:
    action_catalog = _teacher_aux_hand_catalog()
    family_index = family_indices(action_catalog)
    play_action = first_action_id(
        action_catalog,
        family="main_play_character",
        predicate=lambda decoded: decoded.hand_index is not None,
    )
    clock_action = first_action_id(
        action_catalog,
        family="clock_from_hand",
        predicate=lambda decoded: decoded.hand_index is not None,
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    return action_catalog, family_index, play_action, clock_action, play_hand, clock_hand


def test_compute_factorized_teacher_hand_supervision_matches_factorized_branch_hand_terms() -> None:
    action_catalog, family_index, play_action, clock_action, play_hand, clock_hand = _hand_actions()
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[play_action], [clock_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    arg0_logp = torch.tensor([[-0.05], [-0.20]], dtype=torch.float32)
    top_arg0 = torch.tensor([[play_hand], [clock_hand]], dtype=torch.long)
    metadata = structured_catalog_metadata(action_catalog)
    family_log_probs = confident_family_log_probs(action_catalog, ["main_play_character", "clock_from_hand"])
    zero = family_log_probs.sum() * 0.0

    direct = compute_factorized_teacher_hand_supervision(
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        hand_targets_by_action=torch.as_tensor(metadata.hand_indices, dtype=torch.long),
        hand_family_ids=(family_index["main_play_character"], family_index["clock_from_hand"]),
        play_family_id=family_index["main_play_character"],
        clock_from_hand_family_id=family_index["clock_from_hand"],
        hand_coef=1.0,
        zero=zero,
        value_dtype=family_log_probs.dtype,
    )
    factorized_loss, factorized_metrics, _factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=0.4,
        factorized_family_log_probs=family_log_probs,
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
    )

    torch.testing.assert_close(factorized_loss, direct.hand_loss * 0.4)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_hand_targets() -> None:
    action_catalog, family_index, play_action, clock_action, play_hand, clock_hand = _hand_actions()

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[play_action], [clock_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=1.0,
        factorized_family_log_probs=confident_family_log_probs(
            action_catalog,
            ["main_play_character", "clock_from_hand"],
        ),
        factorized_same_family_arg0_logp=torch.tensor([[-0.05], [-0.10]], dtype=torch.float32),
        factorized_same_family_top_arg0=torch.tensor([[play_hand], [clock_hand]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_clock_from_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_hand_loss"] == pytest.approx(0.075)
