from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.plotting.paper_figures import PAPER_FIGURE_IDS, PAPER_FIGURE_STEMS, render_paper_figures


POLICIES = ("alpha", "beta", "gamma")


def _write_run_artifacts(run_dir: Path) -> None:
    _write_payoff_matrix(run_dir)
    _write_truncation_heatmap(run_dir)
    _write_seat_bias_artifact(run_dir)
    _write_training_metrics_artifact(run_dir)


def _write_payoff_matrix(run_dir: Path) -> None:
    _write_matrix(
        run_dir / "eval" / "final_eval" / "payoff_matrices" / "p_mean.csv",
        header=POLICIES,
        rows=(
            ("alpha", (0.50, 0.61, 0.73)),
            ("beta", (0.39, 0.50, 0.57)),
            ("gamma", (0.27, 0.43, 0.50)),
        ),
    )


def _write_truncation_heatmap(run_dir: Path) -> None:
    _write_matrix(
        run_dir / "eval" / "diagnostics" / "truncation_heatmap_data.csv",
        header=POLICIES,
        rows=(
            ("alpha", (0.000, 0.012, 0.020)),
            ("beta", (0.012, 0.000, 0.018)),
            ("gamma", (0.020, 0.018, 0.000)),
        ),
    )


def _write_seat_bias_artifact(run_dir: Path) -> None:
    (run_dir / "eval" / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "diagnostics" / "seat_bias.json").write_text(
        json.dumps(
            {
                "global": {
                    "seat0_win_rate": 0.54,
                    "ci_low": 0.48,
                    "ci_high": 0.60,
                    "decisive_games": 120,
                },
                "matchups": [
                    {
                        "policy_a": "alpha",
                        "policy_b": "beta",
                        "seat0_win_rate": 0.58,
                        "seat1_win_rate": 0.42,
                        "decisive_games": 40,
                    },
                    {
                        "policy_a": "alpha",
                        "policy_b": "gamma",
                        "seat0_win_rate": 0.51,
                        "seat1_win_rate": 0.49,
                        "decisive_games": 40,
                    },
                    {
                        "policy_a": "beta",
                        "policy_b": "gamma",
                        "seat0_win_rate": 0.53,
                        "seat1_win_rate": 0.47,
                        "decisive_games": 40,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_training_metrics_artifact(run_dir: Path) -> None:
    (run_dir / "training" / "logs").mkdir(parents=True, exist_ok=True)
    records = [
        {
            "update_count": 1,
            "wall_clock_seconds": 1.0,
            "wall_clock_ms": 1000,
            "policy_version": 1,
            "loss": 1.8,
            "value_loss": 0.9,
            "actor_loss": 0.7,
            "entropy": 0.4,
            "throughput_samples_per_sec": 128.0,
        },
        {
            "update_count": 2,
            "wall_clock_seconds": 2.0,
            "wall_clock_ms": 2000,
            "policy_version": 2,
            "loss": 1.2,
            "value_loss": 0.7,
            "actor_loss": 0.5,
            "entropy": 0.35,
            "throughput_samples_per_sec": 132.0,
        },
        {
            "update_count": 3,
            "wall_clock_seconds": 3.0,
            "wall_clock_ms": 3000,
            "policy_version": 3,
            "loss": 0.8,
            "value_loss": 0.5,
            "actor_loss": 0.3,
            "entropy": 0.3,
            "throughput_samples_per_sec": 140.0,
        },
    ]
    (run_dir / "training" / "logs" / "training_metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_matrix(path: Path, *, header: tuple[str, ...], rows: tuple[tuple[str, tuple[float, ...]], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["," + ",".join(header)]
    for row_label, values in rows:
        lines.append(row_label + "," + ",".join(f"{value:.3f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_render_paper_figures_writes_all_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "synthetic"
    _write_run_artifacts(run_dir)

    outputs = render_paper_figures(run_dir)

    output_names = {path.name for path in outputs}
    expected_names = {f"{stem}.{fmt}" for stem in PAPER_FIGURE_STEMS for fmt in ("pdf", "png")}
    assert output_names == expected_names
    assert all(path.is_file() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)


def test_render_paper_figures_can_target_single_figure_by_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "seat-bias-only"
    _write_seat_bias_artifact(run_dir)

    outputs = render_paper_figures(run_dir, fig_id="seat_bias")

    assert {path.name for path in outputs} == {"fig_seat_bias.pdf", "fig_seat_bias.png"}
    assert all(path.is_file() for path in outputs)
    assert not (run_dir / "figures" / "paper" / "fig_matchup_heatmap.pdf").exists()


def test_render_paper_figures_rejects_unknown_fig_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown fig_id 'unknown'"):
        render_paper_figures(tmp_path / "runs" / "unknown-figure", fig_id="unknown")

    assert PAPER_FIGURE_IDS == (
        "matchup_heatmap",
        "truncation_heatmap",
        "seat_bias",
        "learning_curves",
    )


def test_render_paper_figures_fails_fast_when_required_artifact_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-seat-bias"
    _write_run_artifacts(run_dir)
    (run_dir / "eval" / "diagnostics" / "seat_bias.json").unlink()

    with pytest.raises(FileNotFoundError, match="seat_bias.json"):
        render_paper_figures(run_dir)


def test_render_paper_figures_checks_only_selected_figure_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-learning-curves"
    _write_run_artifacts(run_dir)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="training_metrics.jsonl"):
        render_paper_figures(run_dir, fig_id="learning_curves")


def test_render_paper_figures_rejects_malformed_heatmap_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "bad-payoff"
    _write_run_artifacts(run_dir)
    (run_dir / "eval" / "final_eval" / "payoff_matrices" / "p_mean.csv").write_text(
        ",alpha,beta\nalpha,0.50,0.55\ngamma,0.45,0.50\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row labels must match the column labels"):
        render_paper_figures(run_dir)


def test_render_paper_figures_rejects_inconsistent_learning_curve_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "bad-training-metrics"
    _write_run_artifacts(run_dir)
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
