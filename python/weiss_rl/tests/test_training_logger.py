from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.vtrace import VtraceMetrics, compute_vtrace_metrics
from weiss_rl.training_logger import TrainingLogger, TrainingMetrics


def _masked_batch(*, time_steps: int = 4, batch_size: int = 2, action_space: int = 5) -> dict[str, np.ndarray]:
    return {
        "logits": np.random.randn(time_steps, batch_size, action_space),
        "behavior_logits": np.random.randn(time_steps, batch_size, action_space),
        "actions": np.random.randint(0, action_space, size=(time_steps, batch_size)),
        "legal_mask": np.ones((time_steps, batch_size, action_space), dtype=bool),
        "rewards": np.random.randn(time_steps, batch_size),
    }


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


def test_vtrace_metrics_ignore_illegal_logits_when_masked() -> None:
    batch = {
        "logits": np.array([[[0.0, 1.0, 100.0, -100.0]]], dtype=np.float32),
        "behavior_logits": np.array([[[0.0, 1.0, -50.0, 50.0]]], dtype=np.float32),
        "actions": np.array([[1]], dtype=np.int64),
        "legal_mask": np.array([[[True, True, False, False]]]),
    }

    metrics = compute_vtrace_metrics(batch)

    assert isinstance(metrics, VtraceMetrics)
    assert metrics.rho_mean == pytest.approx(1.0)
    assert metrics.kl_divergence == pytest.approx(0.0)
    expected_entropy = -(
        (math.e / (math.e + 1.0)) * math.log(math.e / (math.e + 1.0))
        + (1.0 / (math.e + 1.0)) * math.log(1.0 / (math.e + 1.0))
    )
    assert metrics.entropy == pytest.approx(expected_entropy)


def test_vtrace_metrics_support_packed_legal_ids() -> None:
    batch = {
        "logits": np.array([[[1.0, 9.0, 0.0, 9.0]]], dtype=np.float32),
        "behavior_logits": np.array([[[0.0, -9.0, 0.0, -9.0]]], dtype=np.float32),
        "actions": np.array([[2]], dtype=np.int64),
        "legal_ids": np.array([0, 2], dtype=np.int64),
        "legal_offsets": np.array([0, 2], dtype=np.int64),
    }

    metrics = compute_vtrace_metrics(batch)

    expected_rho = (1.0 / (math.e + 1.0)) / 0.5
    expected_entropy = -(
        (math.e / (math.e + 1.0)) * math.log(math.e / (math.e + 1.0))
        + (1.0 / (math.e + 1.0)) * math.log(1.0 / (math.e + 1.0))
    )
    expected_kl = 0.5 * math.log(0.5 / (math.e / (math.e + 1.0))) + 0.5 * math.log(
        0.5 / (1.0 / (math.e + 1.0))
    )

    assert metrics.rho_mean == pytest.approx(expected_rho)
    assert metrics.rho_p90 == pytest.approx(expected_rho)
    assert metrics.kl_divergence == pytest.approx(expected_kl)
    assert metrics.entropy == pytest.approx(expected_entropy)


def test_vtrace_metrics_require_legality_surface() -> None:
    metrics = compute_vtrace_metrics(
        {
            "logits": np.zeros((1, 1, 2), dtype=np.float32),
            "behavior_logits": np.zeros((1, 1, 2), dtype=np.float32),
            "actions": np.zeros((1, 1), dtype=np.int64),
        }
    )

    assert metrics == VtraceMetrics()


def test_impala_learner_logs_masked_metrics_and_uses_update_count_checkpoints(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
        logs_dir=tmp_path / "logs",
        logging_interval_updates=1,
    )

    for _ in range(4):
        result = learner.update(_masked_batch())
        assert set(result) == {
            "loss",
            "throughput_samples_per_sec",
            "throughput_updates_per_sec",
        }

    records = TrainingLogger.read_jsonl(tmp_path / "logs" / "training_metrics.jsonl")
    assert [record["update_count"] for record in records] == [1, 2, 3, 4]
    assert all(record["entropy"] > 0.0 for record in records)
    assert (tmp_path / "checkpoints" / "checkpoint_2.pt").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_4.pt").is_file()
    assert learner.get_policy_version() == 2
