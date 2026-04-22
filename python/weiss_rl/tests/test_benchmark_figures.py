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
    final_eval_payload: dict[str, object] | None = None,
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
    if final_eval_payload is None:
        final_eval_payload = {
            "policy_ids": ["B0 RandomLegal", "policy_000100"],
            "metadata": {"recommended_focal_policy_id": "policy_000100"},
            "matrices": {
                "mean": {
                    "policy_ids": ["B0 RandomLegal", "policy_000100"],
                    "values": [[0.5, 0.0], [final_score, 0.5]],
                }
            },
        }
    (run_dir / "eval" / "final_eval" / "summary.json").write_text(
        json.dumps(final_eval_payload, indent=2, sort_keys=True) + "\n",
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


def test_render_benchmark_figures_uses_recommended_focal_policy_not_best_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "thesis_model_seed1"
    _write_run(
        run_dir,
        algorithm="impala_vtrace_structured_v1",
        recurrent_core="gru",
        final_score=0.60,
        encoder_kind="structured_v2",
        final_eval_payload={
            "policy_ids": ["B0 RandomLegal", "policy_000100", "policy_000200"],
            "metadata": {"recommended_focal_policy_id": "policy_000100"},
            "matrices": {
                "mean": {
                    "policy_ids": ["B0 RandomLegal", "policy_000100", "policy_000200"],
                    "values": [
                        [0.5, 0.0, 0.0],
                        [0.60, 0.5, 0.5],
                        [0.90, 0.5, 0.5],
                    ],
                }
            },
        },
    )

    render_benchmark_figures(run_dirs=[run_dir], out_dir=tmp_path / "figures" / "benchmark")

    summary_payload = json.loads((tmp_path / "figures" / "benchmark" / "benchmark_summary.json").read_text())
    [method] = summary_payload["methods"]
    assert method["method_id"] == "thesis_model"
    assert method["final_score_vs_b0_mean"] == 0.60


def test_render_benchmark_figures_distinguishes_thesis_ablations_by_role(tmp_path: Path) -> None:
    teacher_fade = tmp_path / "runs" / "teacher_fade_seed1"
    no_tactical_bias = tmp_path / "runs" / "no_tactical_bias_seed1"
    _write_run(
        teacher_fade,
        algorithm="impala_vtrace_structured_v1",
        recurrent_core="gru",
        final_score=0.61,
        experiment_role="ablation_teacher_fade",
        encoder_kind="structured_v2",
    )
    _write_run(
        no_tactical_bias,
        algorithm="impala_vtrace_structured_v1",
        recurrent_core="gru",
        final_score=0.57,
        experiment_role="ablation_no_tactical_bias",
        encoder_kind="structured_v2",
    )

    render_benchmark_figures(run_dirs=[teacher_fade, no_tactical_bias], out_dir=tmp_path / "figures" / "benchmark")

    summary_payload = json.loads((tmp_path / "figures" / "benchmark" / "benchmark_summary.json").read_text())
    method_ids = [entry["method_id"] for entry in summary_payload["methods"]]
    assert method_ids == ["thesis_ablation_no_tactical_bias", "thesis_ablation_teacher_fade"]


def test_render_benchmark_figures_distinguishes_noleague_variants_and_treats_b3_b4_as_baselines(tmp_path: Path) -> None:
    canonical = tmp_path / "runs" / "canonical_noleague"
    multideck = tmp_path / "runs" / "multideck_noleague"
    _write_run(
        canonical,
        algorithm="impala_vtrace_structured_v1",
        recurrent_core="gru",
        final_score=0.58,
        experiment_role="baseline_noleague",
        encoder_kind="structured_v2",
    )
    _write_run(
        multideck,
        algorithm="impala_vtrace_structured_v1",
        recurrent_core="gru",
        final_score=0.55,
        experiment_role="baseline_noleague_multideck",
        encoder_kind="structured_v2",
        final_eval_payload={
            "policy_ids": ["B0 RandomLegal", "B3 HeuristicPublicAggro"],
            "metadata": {},
            "matrices": {
                "mean": {
                    "policy_ids": ["B0 RandomLegal", "B3 HeuristicPublicAggro"],
                    "values": [[0.5, 0.0], [0.55, 0.5]],
                }
            },
        },
    )

    render_benchmark_figures(run_dirs=[canonical, multideck], out_dir=tmp_path / "figures" / "benchmark")

    summary_payload = json.loads((tmp_path / "figures" / "benchmark" / "benchmark_summary.json").read_text())
    method_ids = [entry["method_id"] for entry in summary_payload["methods"]]
    assert method_ids == ["impala_no_league", "thesis_multideck_no_league"]
