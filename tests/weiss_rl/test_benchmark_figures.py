from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.plotting.benchmark_figures import render_benchmark_figures


def _write_run(
    run_dir: Path,
    *,
    algorithm: str,
    recurrent_core: str,
    final_score: float,
    experiment_role: str = "main",
    encoder_kind: str = "mlp",
) -> None:
    (run_dir / "training" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "final_eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_canonical.json").write_text(
        json.dumps(
            {
                "config": {
                    "experiment": {"role": experiment_role},
                    "model": {"recurrent_core": recurrent_core, "encoder_kind": encoder_kind},
                    "training": {"algorithm": algorithm},
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "training" / "logs" / "training_metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "update_count": index + 1,
                    "wall_clock_seconds": float(index + 1),
                    "wall_clock_ms": (index + 1) * 1000,
                    "policy_version": index + 1,
                    "loss": 1.0 / float(index + 1),
                    "throughput_samples_per_sec": 100.0 + (10.0 * index),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    (run_dir / "training" / "logs" / "performance.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "kind": "runtime_performance_v1",
                    "wall_clock_seconds": float(index + 1),
                    "actor_env_steps_per_sec": 250.0 + (50.0 * index),
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    (run_dir / "eval" / "final_eval" / "summary.json").write_text(
        json.dumps(
            {
                "policy_ids": ["B0 RandomLegal", "policy_000100"],
                "matrices": {
                    "mean": {
                        "policy_ids": ["B0 RandomLegal", "policy_000100"],
                        "values": [[0.5, 0.0], [final_score, 0.5]],
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_render_benchmark_figures_writes_expected_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "runs" / "impala_main_seed1"
    run_b = tmp_path / "runs" / "impala_main_seed2"
    run_c = tmp_path / "runs" / "ppo_baseline_seed1"
    _write_run(run_a, algorithm="impala_vtrace_gru", recurrent_core="gru", final_score=0.91)
    _write_run(run_b, algorithm="impala_vtrace_gru", recurrent_core="gru", final_score=0.89)
    _write_run(run_c, algorithm="ppo_lite_masked_v1", recurrent_core="gru", final_score=0.73)

    outputs = render_benchmark_figures(
        run_dirs=[run_a, run_b, run_c],
        out_dir=tmp_path / "figures" / "benchmark",
    )

    output_names = {path.name for path in outputs}
    assert "fig_benchmark_loss.png" in output_names
    assert "fig_benchmark_throughput.png" in output_names
    assert "fig_benchmark_final_score.png" in output_names
    assert "benchmark_summary.json" in output_names
    assert "benchmark_summary.csv" in output_names
    assert "benchmark_summary.md" in output_names
    assert "benchmark_runs.csv" in output_names

    summary_payload = json.loads((tmp_path / "figures" / "benchmark" / "benchmark_summary.json").read_text())
    method_ids = [entry["method_id"] for entry in summary_payload["methods"]]
    assert method_ids == ["impala_main", "ppo_lite"]
