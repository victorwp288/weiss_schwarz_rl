from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.plotting.paper_figures import render_paper_figures

from .paper_figures_test_support import write_run_artifacts


def test_render_paper_figures_fails_fast_when_required_artifact_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-seat-bias"
    write_run_artifacts(run_dir)
    (run_dir / "eval" / "diagnostics" / "seat_bias.json").unlink()

    with pytest.raises(FileNotFoundError, match="seat_bias.json"):
        render_paper_figures(run_dir)


def test_render_paper_figures_checks_only_selected_figure_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-learning-curves"
    write_run_artifacts(run_dir)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="training_metrics.jsonl"):
        render_paper_figures(run_dir, fig_id="learning_curves")


def test_render_paper_figures_rejects_malformed_heatmap_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "bad-payoff"
    write_run_artifacts(run_dir)
    (run_dir / "eval" / "final_eval" / "payoff_matrices" / "p_mean.csv").write_text(
        ",alpha,beta\nalpha,0.50,0.55\ngamma,0.45,0.50\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row labels must match the column labels"):
        render_paper_figures(run_dir)


def test_render_paper_figures_rejects_inconsistent_learning_curve_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "bad-training-metrics"
    write_run_artifacts(run_dir)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").write_text(
        "".join(
            (
                json.dumps(
                    {
                        "update_count": 1,
                        "wall_clock_seconds": 1.0,
                        "wall_clock_ms": 1000,
                        "policy_version": 1,
                        "loss": 1.0,
                    },
                    sort_keys=True,
                )
                + "\n",
                json.dumps(
                    {
                        "update_count": 2,
                        "wall_clock_seconds": 2.0,
                        "wall_clock_ms": 2000,
                        "policy_version": 2,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field 'loss' is missing from record 2"):
        render_paper_figures(run_dir)
