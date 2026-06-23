from __future__ import annotations

from pathlib import Path

import pytest

from .learning_progress_test_support import build_overheated_training_summary


def test_learning_progress_diagnostic_summarizes_teacher_guidance(tmp_path: Path) -> None:
    teacher = build_overheated_training_summary(tmp_path)["teacher_guidance"]

    assert teacher["teacher_public_heuristic_coef_active"]["last"] == pytest.approx(0.04)
    assert teacher["teacher_hand_coef_active"]["last"] == pytest.approx(0.08)
    assert teacher["teacher_aux_loss"]["last"] == pytest.approx(0.08)
    assert teacher["teacher_main_play_character_slot_accuracy"]["last"] == pytest.approx(0.55)
    assert teacher["teacher_hand_accuracy"]["last"] == pytest.approx(0.6)
    assert teacher["teacher_main_play_character_hand_accuracy"]["last"] == pytest.approx(0.5)
    assert teacher["teacher_clock_from_hand_accuracy"]["last"] == pytest.approx(0.45)
    assert teacher["teacher_hand_loss"]["last"] == pytest.approx(0.7)
    assert teacher["teacher_hand_supported_fraction"]["last"] == pytest.approx(0.9)
    assert teacher["teacher_same_family_action_accuracy"]["last"] == pytest.approx(0.3)
    assert teacher["teacher_same_family_main_play_character_accuracy"]["last"] == pytest.approx(0.2)
    assert teacher["teacher_action_margin_mean"]["last"] == pytest.approx(0.15)
    assert teacher["teacher_action_margin_satisfied_fraction"]["last"] == pytest.approx(0.25)
    assert teacher["teacher_same_family_action_margin_mean"]["last"] == pytest.approx(0.12)
    assert teacher["teacher_same_family_action_margin_satisfied_fraction"]["last"] == pytest.approx(0.2)
    assert teacher["teacher_public_heuristic_loss"]["last"] == pytest.approx(2.0)
    assert teacher["teacher_public_heuristic_supported_fraction"]["last"] == pytest.approx(0.7)
    assert teacher["teacher_public_heuristic_top1_mass"]["last"] == pytest.approx(0.35)
    assert teacher["teacher_public_heuristic_target_entropy"]["last"] == pytest.approx(1.4)
    assert teacher["teacher_tactical_row_fraction_of_total"]["last"] == pytest.approx(0.15)
    assert teacher["policy_anchor_coef_active"]["last"] == pytest.approx(0.08)
    assert teacher["policy_anchor_loss"]["last"] == pytest.approx(0.12)
    assert teacher["policy_anchor_weighted_loss"]["last"] == pytest.approx(0.0096)
    assert teacher["policy_anchor_kl_mean"]["last"] == pytest.approx(0.12)
    assert teacher["policy_anchor_kl_p95"]["last"] == pytest.approx(0.2)
    assert teacher["policy_anchor_top_action_coef_active"]["last"] == pytest.approx(0.04)
    assert teacher["policy_anchor_top_action_loss"]["last"] == pytest.approx(0.25)
    assert teacher["policy_anchor_top_action_loss_p95"]["last"] == pytest.approx(0.4)
    assert teacher["policy_anchor_top_action_agreement"]["last"] == pytest.approx(0.85)
