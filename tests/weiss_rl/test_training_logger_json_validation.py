from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from weiss_rl.diagnostics.logging.training_logger import TrainingLogger, TrainingMetrics


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
