from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

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


def test_training_logger_sanitizes_nonfinite_values_to_null(tmp_path: Path) -> None:
    logger = TrainingLogger(tmp_path / "logs")
    metrics = TrainingMetrics(
        update_count=1,
        wall_clock_seconds=0.5,
        wall_clock_ms=500,
        policy_version=0,
        vtrace_rho_mean=math.nan,
        kl_divergence=math.inf,
        custom_metrics={"vtrace_batch_metrics_available": 0.0, "vtrace_entropy": math.nan},
    )

    assert math.isnan(metrics.vtrace_rho_mean)
    assert math.isinf(metrics.kl_divergence)
    assert math.isnan(metrics.custom_metrics["vtrace_entropy"])

    logger.log(metrics)

    raw_record = logger.log_file.read_text(encoding="utf-8").strip()
    assert "NaN" not in raw_record
    assert "Infinity" not in raw_record

    record = json.loads(raw_record)
    assert record["vtrace_rho_mean"] is None
    assert record["kl_divergence"] is None
    assert record["custom_metrics"]["vtrace_entropy"] is None


def test_validate_jsonl_rejects_nonfinite_tokens_and_accepts_sanitized_jsonl(tmp_path: Path) -> None:
    invalid_log = tmp_path / "invalid.jsonl"
    invalid_log.write_text(
        '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0,"vtrace_rho_mean":NaN}\n',
        encoding="utf-8",
    )

    is_valid, message = TrainingLogger.validate_jsonl(invalid_log)
    assert not is_valid
    assert "Non-finite float token" in message

    logger = TrainingLogger(tmp_path / "logs")
    logger.log(
        TrainingMetrics(
            update_count=1,
            wall_clock_seconds=0.5,
            wall_clock_ms=500,
            policy_version=0,
            vtrace_rho_mean=math.nan,
        )
    )

    is_valid, message = TrainingLogger.validate_jsonl(logger.log_file)
    assert is_valid
    assert message == "Valid JSONL with 1 records"


def test_validate_jsonl_rejects_missing_required_fields_on_later_records(tmp_path: Path) -> None:
    invalid_log = tmp_path / "invalid.jsonl"
    invalid_log.write_text(
        "\n".join(
            [
                '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0}',
                '{"update_count":2,"wall_clock_seconds":1.0,"wall_clock_ms":1000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    is_valid, message = TrainingLogger.validate_jsonl(invalid_log)
    assert not is_valid
    assert message == "Invalid JSON: Record 2 missing required fields: policy_version"


def test_read_jsonl_rejects_missing_required_fields_on_later_records(tmp_path: Path) -> None:
    invalid_log = tmp_path / "invalid.jsonl"
    invalid_log.write_text(
        "\n".join(
            [
                '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0}',
                '{"update_count":2,"wall_clock_seconds":1.0,"policy_version":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Record 2 missing required fields: wall_clock_ms"):
        TrainingLogger.read_jsonl(invalid_log)


@pytest.mark.parametrize(
    ("record_json", "message"),
    [
        (
            '{"update_count":null,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0}\n',
            "Record 1 field update_count must be an integer",
        ),
        (
            '{"update_count":true,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0}\n',
            "Record 1 field update_count must be an integer",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":null,"wall_clock_ms":500,"policy_version":0}\n',
            "Record 1 field wall_clock_seconds must be a finite number",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":true,"wall_clock_ms":500,"policy_version":0}\n',
            "Record 1 field wall_clock_seconds must be a finite number",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":"0.5","wall_clock_ms":500,"policy_version":0}\n',
            "Record 1 field wall_clock_seconds must be a finite number",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":null,"policy_version":0}\n',
            "Record 1 field wall_clock_ms must be an integer",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":false,"policy_version":0}\n',
            "Record 1 field wall_clock_ms must be an integer",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":null}\n',
            "Record 1 field policy_version must be an integer",
        ),
        (
            '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":true}\n',
            "Record 1 field policy_version must be an integer",
        ),
    ],
)
def test_validate_jsonl_rejects_invalid_required_field_types(tmp_path: Path, record_json: str, message: str) -> None:
    invalid_log = tmp_path / "invalid_types.jsonl"
    invalid_log.write_text(record_json, encoding="utf-8")

    is_valid, actual_message = TrainingLogger.validate_jsonl(invalid_log)
    assert not is_valid
    assert actual_message == f"Invalid JSON: {message}"


def test_read_jsonl_allows_null_optional_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "optional_nulls.jsonl"
    log_path.write_text(
        '{"update_count":1,"wall_clock_seconds":0.5,"wall_clock_ms":500,"policy_version":0,"vtrace_rho_mean":null,"custom_metrics":{"vtrace_entropy":null}}\n',
        encoding="utf-8",
    )

    assert TrainingLogger.read_jsonl(log_path) == [
        {
            "update_count": 1,
            "wall_clock_seconds": 0.5,
            "wall_clock_ms": 500,
            "policy_version": 0,
            "vtrace_rho_mean": None,
            "custom_metrics": {"vtrace_entropy": None},
        }
    ]


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
    expected_kl = 0.5 * math.log(0.5 / (math.e / (math.e + 1.0))) + 0.5 * math.log(0.5 / (1.0 / (math.e + 1.0)))

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

    assert math.isnan(metrics.rho_mean)
    assert math.isnan(metrics.rho_p50)
    assert math.isnan(metrics.rho_p90)
    assert math.isnan(metrics.rho_p99)
    assert math.isnan(metrics.clip_rate)
    assert math.isnan(metrics.c_clipped_rate)
    assert math.isnan(metrics.kl_divergence)
    assert math.isnan(metrics.entropy)


def test_vtrace_metrics_support_object_batches() -> None:
    batch = SimpleNamespace(**_masked_batch(time_steps=1, batch_size=1, action_space=3))

    metrics = compute_vtrace_metrics(batch)

    assert isinstance(metrics, VtraceMetrics)
    assert math.isfinite(metrics.rho_mean)
    assert math.isfinite(metrics.kl_divergence)
    assert math.isfinite(metrics.entropy)


def test_impala_learner_logs_masked_metrics_and_uses_update_count_checkpoint_metadata(tmp_path: Path) -> None:
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
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_2.json").is_file()
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_logs_vtrace_metrics_for_object_batches(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        logs_dir=tmp_path / "logs",
        logging_interval_updates=1,
    )

    result = learner.update(SimpleNamespace(**_masked_batch(time_steps=1, batch_size=1, action_space=3)))

    assert set(result) == {
        "loss",
        "throughput_samples_per_sec",
        "throughput_updates_per_sec",
    }

    [record] = TrainingLogger.read_jsonl(tmp_path / "logs" / "training_metrics.jsonl")
    assert math.isfinite(record["vtrace_rho_mean"])
    assert math.isfinite(record["kl_divergence"])
    assert math.isfinite(record["entropy"])
    assert record["custom_metrics"]["vtrace_batch_metrics_available"] == 1.0
    assert record["custom_metrics"]["vtrace_entropy"] == pytest.approx(record["entropy"])
