from __future__ import annotations

import math
from pathlib import Path

import pytest
from weiss_rl.diagnostics.training_logger import TrainingLogger, TrainingMetrics


def test_merge_latest_custom_metrics_updates_single_training_row(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs")
    logger.log(
        TrainingMetrics(
            update_count=3,
            wall_clock_seconds=0.5,
            wall_clock_ms=500,
            policy_version=2,
            custom_metrics={"vtrace_batch_metrics_available": 1.0},
        )
    )

    record = logger.merge_latest_custom_metrics(
        update_count=3,
        policy_version=2,
        metrics={
            "paired_swing_replay_loss": 0.25,
            "trajectory_bc_replay_aux_updates": 1,
            "unrelated_metric": 99.0,
            "paired_swing_replay_nan": math.nan,
        },
    )

    assert record is not None
    [saved] = TrainingLogger.read_jsonl(logger.log_file)
    assert saved["update_count"] == 3
    assert saved["custom_metrics"]["vtrace_batch_metrics_available"] == 1.0
    assert saved["custom_metrics"]["paired_swing_replay_loss"] == pytest.approx(0.25)
    assert saved["custom_metrics"]["trajectory_bc_replay_aux_updates"] == pytest.approx(1.0)
    assert "unrelated_metric" not in saved["custom_metrics"]
    assert "paired_swing_replay_nan" not in saved["custom_metrics"]


def test_merge_latest_custom_metrics_can_include_pfsp_runtime_metrics(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs")
    logger.log(
        TrainingMetrics(
            update_count=4,
            wall_clock_seconds=0.5,
            wall_clock_ms=500,
            policy_version=2,
        )
    )

    record = logger.merge_latest_custom_metrics(
        update_count=4,
        policy_version=2,
        metrics={
            "pfsp_champion_envs": 68,
            "pfsp_hard_negative_envs": 90,
            "collector_pfsp_sampled_envs": 764,
            "unrelated_metric": 99.0,
        },
        prefixes=("pfsp_", "collector_pfsp_"),
    )

    assert record is not None
    [saved] = TrainingLogger.read_jsonl(logger.log_file)
    assert saved["custom_metrics"]["pfsp_champion_envs"] == pytest.approx(68.0)
    assert saved["custom_metrics"]["pfsp_hard_negative_envs"] == pytest.approx(90.0)
    assert saved["custom_metrics"]["collector_pfsp_sampled_envs"] == pytest.approx(764.0)
    assert "unrelated_metric" not in saved["custom_metrics"]


def test_merge_latest_custom_metrics_rejects_wrong_latest_update(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs")
    logger.log(
        TrainingMetrics(
            update_count=3,
            wall_clock_seconds=0.5,
            wall_clock_ms=500,
            policy_version=2,
        )
    )

    with pytest.raises(ValueError, match="does not match post-update metrics"):
        logger.merge_latest_custom_metrics(
            update_count=4,
            policy_version=2,
            metrics={"paired_swing_replay_loss": 0.25},
        )
