from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.diagnostics.learning_progress import build_learning_progress_summary

from .learning_progress_test_support import write_promotion_gate_failure_fixture


def test_learning_progress_diagnostic_summarizes_promotion_gate_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_promotion_gate_failure_fixture(run_dir)

    summary = build_learning_progress_summary(run_dir)
    gate = summary["promotion_gate"]

    assert gate["attempt_count"] == 3
    assert gate["passed_count"] == 0
    assert gate["failed_count"] == 3
    assert gate["first_pass_update"] is None
    assert gate["latest_update"] == 20
    assert gate["latest_passed"] is False
    assert gate["latest_reason_codes"] == ["anchor_loss_guardrail_exceeded"]
    assert gate["consecutive_failure_count"] == 3
    assert gate["records"][0]["anchor_means"]["B4 HeuristicPublicControl"] == pytest.approx(0.375)
    assert summary["league_sampling"]["latest_has_admitted_champion"] is False
    assert summary["league_sampling"]["latest_probationary_recent_sampling_active"] is True
    assert summary["league_sampling"]["snapshot_env_fraction"]["last"] == pytest.approx(0.4)
    assert any("promotion gate never passed" in warning for warning in summary["warnings"])
    assert any("probationary snapshot sampling was active" in warning for warning in summary["warnings"])
    assert any("promotion gate failed 3 consecutive attempts" in warning for warning in summary["warnings"])
