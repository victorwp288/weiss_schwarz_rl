from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.diagnostics.learning_progress import build_learning_progress_summary

from .learning_progress_test_support import write_periodic_dev_eval_trend_fixture


def test_learning_progress_diagnostic_summarizes_periodic_dev_eval_trend(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_periodic_dev_eval_trend_fixture(run_dir)

    summary = build_learning_progress_summary(run_dir)

    trend = summary["periodic_dev_eval"]
    assert trend["best_update"] == 50
    assert trend["best_aggregate_score"] == 0.60
    assert trend["last_update"] == 75
    assert trend["last_aggregate_score"] == 0.52
    assert trend["latest_minus_best"] == pytest.approx(-0.08)
    assert trend["non_monotonic_drop_count"] == 1
    assert summary["actor_model_sync"]["max_policy_version_lag_p90"] == 0.0
    assert summary["actor_model_sync"]["max_learner_actor_update_lag_p90"] == 49.0
    assert summary["actor_model_sync"]["max_learner_to_actor_update_lag"] == 49.0
    assert summary["league_sync"]["max_league_update_lag"] == 49.0
    assert summary["off_policy"]["stale_policy_lag_source"] == "learner_actor_update_lag_p90"
    assert summary["off_policy"]["stale_policy_lag_correlations"]["vtrace_rho_p99"]["paired_update_count"] == 3
    assert summary["off_policy"]["stale_policy_lag_correlations"]["vtrace_rho_p99"]["pearson"] == pytest.approx(
        0.94491118
    )
    assert any("latest periodic dev-eval aggregate" in warning for warning in summary["warnings"])
    assert any("learner_actor_update_lag_p90 exceeded" in warning for warning in summary["warnings"])
