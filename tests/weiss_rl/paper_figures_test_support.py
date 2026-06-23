from __future__ import annotations

import json
from pathlib import Path

POLICIES = ("alpha", "beta", "gamma")


def write_run_artifacts(run_dir: Path) -> None:
    write_payoff_matrix(run_dir)
    write_truncation_heatmap(run_dir)
    write_seat_bias_artifact(run_dir)
    write_training_metrics_artifact(run_dir)


def write_payoff_matrix(run_dir: Path) -> None:
    write_matrix(
        run_dir / "eval" / "final_eval" / "payoff_matrices" / "p_mean.csv",
        header=POLICIES,
        rows=(
            ("alpha", (0.50, 0.61, 0.73)),
            ("beta", (0.39, 0.50, 0.57)),
            ("gamma", (0.27, 0.43, 0.50)),
        ),
    )


def write_truncation_heatmap(run_dir: Path) -> None:
    write_matrix(
        run_dir / "eval" / "diagnostics" / "truncation_heatmap_data.csv",
        header=POLICIES,
        rows=(
            ("alpha", (0.000, 0.012, 0.020)),
            ("beta", (0.012, 0.000, 0.018)),
            ("gamma", (0.020, 0.018, 0.000)),
        ),
    )


def write_seat_bias_artifact(run_dir: Path) -> None:
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


def write_training_metrics_artifact(run_dir: Path) -> None:
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


def write_matrix(path: Path, *, header: tuple[str, ...], rows: tuple[tuple[str, tuple[float, ...]], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["," + ",".join(header)]
    for row_label, values in rows:
        lines.append(row_label + "," + ",".join(f"{value:.3f}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
