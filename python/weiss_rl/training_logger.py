"""Training metrics logger for JSONL export."""

from __future__ import annotations

import json
import time
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
            f.write(json.dumps(record) + "\n")

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
            f.write(json.dumps(metrics_dict) + "\n")

    @staticmethod
    def read_jsonl(log_path: Path) -> list[dict[str, Any]]:
        """Read and parse JSONL log file.
        
        Args:
            log_path: Path to JSONL log file.
            
        Returns:
            List of parsed JSON objects.
        """
        records = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
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
            records = TrainingLogger.read_jsonl(log_path)
            if not records:
                return False, "Log file is empty"
            
            # Check required fields in first record
            first = records[0]
            required_fields = {"update_count", "wall_clock_seconds", "wall_clock_ms", "policy_version"}
            if not required_fields.issubset(first.keys()):
                missing = required_fields - set(first.keys())
                return False, f"Missing required fields: {missing}"
            
            # Validate all records have update_count
            for i, record in enumerate(records):
                if "update_count" not in record:
                    return False, f"Record {i} missing update_count"
            
            return True, f"Valid JSONL with {len(records)} records"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Error reading log: {e}"
