from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from weiss_rl.tensorboard_logger import TensorBoardLogger


def _load_accumulator(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"images": 0, "scalars": 0, "tensors": 0})
    accumulator.Reload()
    return accumulator


def test_tensorboard_logger_records_training_checkpoint_and_dev_eval_metrics(tmp_path: Path) -> None:
    logger = TensorBoardLogger(tmp_path / "tensorboard")
    logger.log_run_context(
        manifest={
            "run_id256": "ab" * 32,
            "config_canonical": {"training": {"optimizer": {"learning_rate": 2e-4}}},
            "simulator": {"version": "0.7.0"},
            "hardware": {"learner_device": "cuda"},
            "evaluation_pinning": {"eval_device": "cpu"},
            "policy_set_selection": ["policy_000001"],
        },
        environment={"python": {"version": "3.12.10"}},
        run_summary={"runtime_mode": "train_ordered"},
        determinism_report={"policy_selection_mode": "deterministic_v1"},
    )
    logger.log_training_step(
        update_count=3,
        policy_version=1,
        wall_clock_seconds=12.5,
        metrics={
            "loss": 0.25,
            "policy_loss": 0.2,
            "value_loss": 0.1,
            "entropy": 1.3,
            "throughput_samples_per_sec": 123.0,
            "throughput_updates_per_sec": 7.0,
            "vtrace_rho_mean": 0.98,
            "actor_env_steps_per_sec": 456.0,
            "queue_occupancy_p50": 0.2,
            "snapshot_publish_latency_ms": 11.0,
            "pfsp_pool_size": 4.0,
            "approx_kl": 0.03,
            "clip_fraction": 0.15,
        },
    )
    logger.log_checkpoint_tracker(
        {
            "latest": {
                "update_count": 3,
                "policy_version": 1,
                "metric_kind": "training_loss",
                "metric_value": 0.25,
            },
            "best": {
                "update_count": 3,
                "policy_version": 1,
                "metric_kind": "dev_eval_mean",
                "metric_value": 0.61,
            },
        },
        step=3,
    )
    logger.log_checkpoint_tracker(
        {
            "latest": {
                "update_count": 3,
                "policy_version": 1,
                "metric_kind": "training_loss",
                "metric_value": 0.25,
            },
            "best": {
                "update_count": 3,
                "policy_version": 1,
                "metric_kind": "dev_eval_mean",
                "metric_value": 0.61,
            },
        },
        step=3,
    )
    logger.log_periodic_dev_eval(
        {
            "summary": {"games": 8, "wins": 5, "losses": 3},
            "uncertainty": {"mean": 0.625, "ci_low": 0.51, "ci_high": 0.73},
            "stop_reason": "precision",
        },
        step=3,
    )
    logger.close()

    accumulator = _load_accumulator(tmp_path / "tensorboard")
    scalar_tags = set(accumulator.Tags()["scalars"])
    tensor_tags = set(accumulator.Tags()["tensors"])

    assert "train/loss" in scalar_tags
    assert "train/policy_loss" in scalar_tags
    assert "throughput/samples_per_sec" in scalar_tags
    assert "vtrace/rho_mean" in scalar_tags
    assert "runtime/actor_env_steps_per_sec" in scalar_tags
    assert "runtime/queue_occupancy_p50" in scalar_tags
    assert "runtime/snapshot_publish_latency_ms" in scalar_tags
    assert "league/pfsp_pool_size" in scalar_tags
    assert "checkpoint/latest/update_count" in scalar_tags
    assert "checkpoint/best/metric_value" in scalar_tags
    assert "dev_eval/uncertainty/mean" in scalar_tags
    assert len(accumulator.Scalars("checkpoint/latest/update_count")) == 1
    assert len(accumulator.Scalars("checkpoint/best/metric_value")) == 1
    assert "run/manifest/text_summary" in tensor_tags
    assert "checkpoint/tracker/text_summary" in tensor_tags
    assert "dev_eval/summary/text_summary" in tensor_tags


def test_tensorboard_logger_records_eval_summaries_and_figures(tmp_path: Path) -> None:
    metagame_dir = tmp_path / "eval" / "metagame"
    (metagame_dir / "S0" / "nash").mkdir(parents=True, exist_ok=True)
    (metagame_dir / "S0" / "alpharank").mkdir(parents=True, exist_ok=True)
    (metagame_dir / "S0" / "nash" / "mixture_mean.csv").write_text(
        "policy_id,mean_mixture\nb0_randomlegal,0.25\npolicy_000001,0.75\n",
        encoding="utf-8",
    )
    (metagame_dir / "S0" / "alpharank" / "stationary_mean.csv").write_text(
        "policy_id,mean_stationary_mass\nb0_randomlegal,0.4\npolicy_000001,0.6\n",
        encoding="utf-8",
    )

    logger = TensorBoardLogger(tmp_path / "tensorboard")
    logger.log_final_eval_summary(
        {
            "policy_ids": ["b0_randomlegal", "policy_000001"],
            "matchups": [{"id": 1}, {"id": 2}, {"id": 3}],
            "posterior_samples": {"sample_count": 1000},
            "matrices": {
                "mean": {
                    "policy_ids": ["b0_randomlegal", "policy_000001"],
                    "values": [[0.5, 0.1], [0.9, 0.5]],
                },
                "games": {
                    "policy_ids": ["b0_randomlegal", "policy_000001"],
                    "values": [[2, 2], [2, 2]],
                },
            },
        }
    )
    logger.log_metagame_summary(
        {
            "policy_ids": ["b0_randomlegal", "policy_000001"],
            "sample_count": 1000,
        },
        metagame_dir=metagame_dir,
    )
    logger.log_paper_readiness(
        {
            "passed": False,
            "alarms": ["baseline_win_rate_vs_b0"],
            "checks": {
                "baseline_win_rate_vs_b0": {"passed": False, "prob_gt_threshold": 0.0},
                "truncation_rate": {"passed": True, "rate": 0.0},
            },
        }
    )
    logger.close()

    accumulator = _load_accumulator(tmp_path / "tensorboard")
    scalar_tags = set(accumulator.Tags()["scalars"])
    image_tags = set(accumulator.Tags()["images"])
    tensor_tags = set(accumulator.Tags()["tensors"])

    assert "eval/final/policy_count" in scalar_tags
    assert "eval/final/mean/b0_randomlegal__vs__policy_000001" in scalar_tags
    assert "eval/metagame/nash_mixture/policy_000001" in scalar_tags
    assert "eval/metagame/alpharank_stationary/b0_randomlegal" in scalar_tags
    assert "eval/readiness/passed" in scalar_tags
    assert "eval/readiness/checks/baseline_win_rate_vs_b0/prob_gt_threshold" in scalar_tags
    assert "eval/final/summary/text_summary" in tensor_tags
    assert "eval/metagame/summary/text_summary" in tensor_tags
    assert "eval/readiness/summary/text_summary" in tensor_tags
    assert "eval/final/mean_heatmap" in image_tags
    assert "eval/metagame/nash_mixture_bar" in image_tags
