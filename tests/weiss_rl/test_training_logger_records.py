from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.diagnostics.logging.training_logger import TrainingLogger, TrainingMetrics


def test_training_metrics_dataclass_defaults() -> None:
    metrics = TrainingMetrics(
        update_count=100,
        wall_clock_seconds=10.5,
        wall_clock_ms=10500,
        policy_version=2,
    )

    assert metrics.update_count == 100
    assert metrics.policy_version == 2
    assert metrics.checkpoint_lag_updates == 0
    assert metrics.custom_metrics == {}


def test_training_logger_roundtrip_and_validation(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs")
    logger.log(
        TrainingMetrics(
            update_count=1,
            wall_clock_seconds=0.5,
            wall_clock_ms=500,
            policy_version=0,
            loss=0.25,
        )
    )

    records = TrainingLogger.read_jsonl(logger.log_file)
    assert records == [
        {
            "update_count": 1,
            "wall_clock_seconds": 0.5,
            "wall_clock_ms": 500,
            "policy_version": 0,
            "loss": 0.25,
            "throughput_samples_per_sec": 0.0,
            "throughput_updates_per_sec": 0.0,
            "vtrace_rho_mean": 0.0,
            "vtrace_rho_p50": 0.0,
            "vtrace_rho_p90": 0.0,
            "vtrace_rho_p99": 0.0,
            "vtrace_clip_rate": 0.0,
            "vtrace_c_clipped_rate": 0.0,
            "kl_divergence": 0.0,
            "checkpoint_lag_updates": 0,
            "checkpoint_lag_percentile_p50": 0.0,
            "checkpoint_lag_percentile_p90": 0.0,
            "value_loss": 0.0,
            "actor_loss": 0.0,
            "entropy": 0.0,
        }
    ]

    is_valid, message = TrainingLogger.validate_jsonl(logger.log_file)
    assert is_valid
    assert message == "Valid JSONL with 1 records"


def test_log_dict_requires_core_fields_and_fills_clock_fields(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs", start_time=0.0)

    with pytest.raises(ValueError, match="update_count"):
        logger.log_dict({"policy_version": 0})
    with pytest.raises(ValueError, match="policy_version"):
        logger.log_dict({"update_count": 1})

    logger.log_dict({"update_count": 3, "policy_version": 1, "loss": 0.5})
    record = TrainingLogger.read_jsonl(logger.log_file)[0]
    assert record["update_count"] == 3
    assert record["policy_version"] == 1
    assert "wall_clock_seconds" in record
    assert record["wall_clock_ms"] >= 0
