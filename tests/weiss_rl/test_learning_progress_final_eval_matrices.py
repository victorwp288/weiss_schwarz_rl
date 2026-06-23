from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.diagnostics.learning_progress import build_learning_progress_summary

from .learning_progress_test_support import write_final_eval_matrix_fixture


def test_learning_progress_diagnostic_summarizes_generic_final_eval_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_final_eval_matrix_fixture(run_dir)

    summary = build_learning_progress_summary(run_dir)

    matrix = summary["final_eval_matrix"]
    assert matrix["mean"]["row_policy_ids"] == ["policy_000002", "policy_000003", "policy_000004"]
    assert matrix["wins"]["values"][2] == [36.0, 34.0, 32.0]
    assert matrix["checkpoint_best_policy_id"] == "policy_000002"
    assert matrix["checkpoint_best_row_mean_excluding_self"] == pytest.approx(0.485)
    assert matrix["best_row_policy_id"] == "policy_000004"
    assert matrix["best_row_update"] == 100
    assert matrix["best_row_mean_excluding_self"] == pytest.approx(0.545)
    matrices = summary["final_eval_matrices"]
    assert sorted(matrices) == ["final_eval", "final_eval_argmax_probe"]
    assert matrices["final_eval_argmax_probe"]["checkpoint_best_policy_id"] == "policy_000002"
    assert matrices["final_eval_argmax_probe"]["checkpoint_best_row_mean_excluding_self"] == pytest.approx(0.65)
    assert matrices["final_eval_argmax_probe"]["wins"]["values"][0] == [32.0, 45.0, 38.0]
    assert any("not the strongest row" in warning for warning in summary["warnings"])
