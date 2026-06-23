from __future__ import annotations

from pathlib import Path

import pytest

from .learning_progress_test_support import build_overheated_training_summary


def test_learning_progress_diagnostic_summarizes_reward_scale_and_chosen_actions(tmp_path: Path) -> None:
    summary = build_overheated_training_summary(tmp_path)

    assert summary["reward_scale"]["reward_abs_mean"]["last"] == 0.25
    assert summary["reward_scale"]["reward_positive_fraction"]["last"] == 0.3
    assert summary["reward_scale"]["reward_negative_fraction"]["last"] == 0.4
    assert summary["reward_scale"]["max_reward_abs_mean"] == 0.25
    assert summary["reward_scale"]["max_target_abs_mean"] == 0.8
    assert summary["chosen_action_learning"]["chosen_pass_train_fraction"]["last"] == pytest.approx(0.75)
    assert summary["chosen_action_learning"]["chosen_pass_train_advantage_mean"]["last"] == pytest.approx(-0.2)
    assert summary["chosen_action_learning"]["chosen_nonpass_train_advantage_mean"]["last"] == pytest.approx(0.4)
    assert summary["chosen_action_learning"]["chosen_mulligan_confirm_train_fraction"]["last"] == pytest.approx(0.01)
    assert summary["chosen_action_learning"]["chosen_mulligan_select_train_fraction"]["last"] == pytest.approx(0.09)
    assert summary["chosen_action_learning"]["chosen_mulligan_select_share_of_mulligan"]["last"] == pytest.approx(0.9)
    assert summary["chosen_action_learning"]["chosen_mulligan_confirm_train_advantage_mean"]["last"] == pytest.approx(
        0.2
    )
    assert summary["chosen_action_learning"]["chosen_mulligan_select_train_advantage_mean"]["last"] == pytest.approx(
        -0.4
    )
    assert any("mulligan-confirm collapse suspected" in warning for warning in summary["warnings"])
