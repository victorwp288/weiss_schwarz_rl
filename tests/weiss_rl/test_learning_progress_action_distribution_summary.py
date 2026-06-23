from __future__ import annotations

from pathlib import Path

import pytest

from .learning_progress_test_support import build_action_distribution_summary


def test_learning_progress_diagnostic_summarizes_action_distribution(tmp_path: Path) -> None:
    summary = build_action_distribution_summary(tmp_path)

    actions = summary["action_distribution"]
    assert actions["main_move_fraction"]["first"] == pytest.approx(0.07)
    assert actions["main_move_fraction"]["last"] == pytest.approx(0.10)
    assert actions["pass_fraction"]["last"] == pytest.approx(0.40)
    assert actions["pass_with_nonpass_fraction_of_total"]["last"] == pytest.approx(0.20)
    assert actions["pass_with_nonpass_fraction_of_pass"]["last"] == pytest.approx(0.50)
    assert actions["mulligan_select_with_confirm_penalty_fraction_of_total"]["last"] == pytest.approx(0.05)
    assert actions["main_move_only_force_pass_rows_fraction_of_total"]["last"] == pytest.approx(0.03)
    assert actions["main_move_only_force_pass_actions_fraction_of_total"]["last"] == pytest.approx(0.07)
    assert actions["max_consecutive_main_moves"]["last"] == pytest.approx(2.0)
    assert actions["max_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert any("collector_max_consecutive_main_moves exceeded" in warning for warning in summary["warnings"])
