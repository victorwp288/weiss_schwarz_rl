"""Training metrics logger for JSONL export."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingMetrics:
    """Training metrics for a single update."""

    update_count: int
    wall_clock_seconds: float
    wall_clock_ms: int
    policy_version: int
    loss: float = 0.0

    # Throughput metrics
    throughput_samples_per_sec: float = 0.0
    throughput_updates_per_sec: float = 0.0

    # V-trace health metrics
    vtrace_rho_mean: float = 0.0
    vtrace_rho_p50: float = 0.0
    vtrace_rho_p90: float = 0.0
    vtrace_rho_p99: float = 0.0

    vtrace_clip_rate: float = 0.0
    vtrace_c_clipped_rate: float = 0.0

    # Policy divergence
    kl_divergence: float = 0.0

    # Optional actor sync lag metrics from checkpoint-based sync.
    checkpoint_lag_updates: int = 0
    checkpoint_lag_percentile_p50: float = 0.0
    checkpoint_lag_percentile_p90: float = 0.0

    # Additional health indicators
    value_loss: float = 0.0
    actor_loss: float = 0.0
    entropy: float = 0.0

    # Custom metrics from learner
    custom_metrics: dict[str, float] = field(default_factory=dict)


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    return value


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ValueError(f"Non-finite float token is not valid JSON: {token}")


REQUIRED_JSONL_FIELDS = frozenset({"update_count", "wall_clock_seconds", "wall_clock_ms", "policy_version"})


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_jsonl_record(line: str, *, line_number: int) -> dict[str, Any]:
    record = json.loads(line, parse_constant=_reject_nonfinite_json_constant)
    if not isinstance(record, dict):
        raise ValueError(f"Record {line_number} is not a JSON object")

    missing_fields = REQUIRED_JSONL_FIELDS - record.keys()
    if missing_fields:
        missing_list = ", ".join(sorted(missing_fields))
        raise ValueError(f"Record {line_number} missing required fields: {missing_list}")

    if not _is_strict_int(record["update_count"]):
        raise ValueError(f"Record {line_number} field update_count must be an integer")
    if not _is_strict_finite_number(record["wall_clock_seconds"]):
        raise ValueError(f"Record {line_number} field wall_clock_seconds must be a finite number")
    if not _is_strict_int(record["wall_clock_ms"]):
        raise ValueError(f"Record {line_number} field wall_clock_ms must be an integer")
    if not _is_strict_int(record["policy_version"]):
        raise ValueError(f"Record {line_number} field policy_version must be an integer")

    return record


class TrainingLogger:
    """Structured JSONL logger for training metrics."""

    def __init__(self, logs_dir: Path, start_time: float | None = None):
        """Initialize logger.

        Args:
            logs_dir: Directory to write logs (should be runs/{run_dir}/logs/)
            start_time: Wall clock start time for computing elapsed time.
                If None, the logger captures the current wall clock time at init.
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.logs_dir / "training_metrics.jsonl"
        self.start_time = time.time() if start_time is None else start_time

    def log(self, metrics: TrainingMetrics) -> None:
        """Log a metrics record to JSONL.

        Args:
            metrics: TrainingMetrics instance to log.
        """
        record = asdict(metrics)

        # Remove custom_metrics if empty
        if not record.get("custom_metrics"):
            del record["custom_metrics"]

        # Write as JSONL (one json object per line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize_json_value(record), allow_nan=False) + "\n")

    def log_dict(self, metrics_dict: dict[str, Any]) -> None:
        """Log a raw metrics dictionary."""
        if "update_count" not in metrics_dict:
            raise ValueError("metrics_dict must include update_count")
        if "policy_version" not in metrics_dict:
            raise ValueError("metrics_dict must include policy_version")
        if "wall_clock_seconds" not in metrics_dict:
            metrics_dict["wall_clock_seconds"] = time.time() - self.start_time
        if "wall_clock_ms" not in metrics_dict:
            metrics_dict["wall_clock_ms"] = int(metrics_dict["wall_clock_seconds"] * 1000)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize_json_value(metrics_dict), allow_nan=False) + "\n")

    def merge_latest_custom_metrics(
        self,
        *,
        update_count: int,
        policy_version: int,
        metrics: Mapping[str, Any],
        prefixes: Sequence[str] = ("trajectory_bc_replay_", "paired_swing_replay_"),
    ) -> dict[str, Any] | None:
        """Merge post-update auxiliary metrics into the latest JSONL record.

        Learner updates write ``training_metrics.jsonl`` before optional replay
        auxiliaries run. This helper keeps the canonical artifact to one row per
        update while still making those post-update auxiliary metrics visible.
        """

        return merge_latest_training_custom_metrics(
            self.log_file,
            update_count=update_count,
            policy_version=policy_version,
            metrics=metrics,
            prefixes=prefixes,
        )

    @staticmethod
    def read_jsonl(log_path: Path) -> list[dict[str, Any]]:
        """Read and parse JSONL log file.

        Args:
            log_path: Path to JSONL log file.

        Returns:
            List of parsed JSON objects.
        """
        records = []
        with open(log_path, encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if line:
                    records.append(_parse_jsonl_record(line, line_number=line_number))
        return records

    @staticmethod
    def validate_jsonl(log_path: Path) -> tuple[bool, str]:
        """Validate JSONL file structure.

        Args:
            log_path: Path to JSONL log file.

        Returns:
            Tuple of (is_valid, message).
        """
        if not log_path.exists():
            return False, f"Log file does not exist: {log_path}"

        try:
            record_count = 0
            with open(log_path, encoding="utf-8") as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    _parse_jsonl_record(line, line_number=line_number)
                    record_count += 1

            if record_count == 0:
                return False, "Log file is empty"

            return True, f"Valid JSONL with {record_count} records"
        except (json.JSONDecodeError, ValueError) as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Error reading log: {e}"


def merge_latest_training_custom_metrics(
    log_path: Path,
    *,
    update_count: int,
    policy_version: int,
    metrics: Mapping[str, Any],
    prefixes: Sequence[str] = ("trajectory_bc_replay_", "paired_swing_replay_"),
) -> dict[str, Any] | None:
    """Merge finite custom metrics into the latest training JSONL record."""

    patch = _custom_metric_patch(metrics, prefixes=prefixes)
    if not patch:
        return None
    if not log_path.is_file():
        return None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    latest_index = _latest_nonempty_line_index(lines)
    if latest_index is None:
        return None
    record = _parse_jsonl_record(lines[latest_index].strip(), line_number=latest_index + 1)
    if int(record["update_count"]) != int(update_count) or int(record["policy_version"]) != int(policy_version):
        raise ValueError(
            "latest training metrics record does not match post-update metrics: "
            f"record update/policy={record['update_count']}/{record['policy_version']} "
            f"requested={int(update_count)}/{int(policy_version)}"
        )
    custom_metrics = record.setdefault("custom_metrics", {})
    if not isinstance(custom_metrics, dict):
        raise ValueError("latest training metrics record has non-object custom_metrics")
    custom_metrics.update(patch)
    lines[latest_index] = json.dumps(_sanitize_json_value(record), allow_nan=False)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def _custom_metric_patch(metrics: Mapping[str, Any], *, prefixes: Sequence[str]) -> dict[str, float]:
    normalized_prefixes = tuple(str(prefix) for prefix in prefixes)
    patch: dict[str, float] = {}
    for key, value in metrics.items():
        if normalized_prefixes and not any(str(key).startswith(prefix) for prefix in normalized_prefixes):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            patch[str(key)] = numeric_value
    return patch


def _latest_nonempty_line_index(lines: Sequence[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if str(lines[index]).strip():
            return index
    return None
