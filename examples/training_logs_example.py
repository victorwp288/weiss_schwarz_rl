"""Standalone example for learner-side training logs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.training_logger import TrainingLogger


def _example_batch(*, time_steps: int = 8, batch_size: int = 2, action_space: int = 5) -> dict[str, np.ndarray]:
    return {
        "logits": np.random.randn(time_steps, batch_size, action_space),
        "behavior_logits": np.random.randn(time_steps, batch_size, action_space),
        "actions": np.random.randint(0, action_space, size=(time_steps, batch_size)),
        "legal_mask": np.ones((time_steps, batch_size, action_space), dtype=bool),
        "rewards": np.random.randn(time_steps, batch_size),
    }


def example_training_loop() -> None:
    run_dir = Path("runs/example_run")
    learner = ImpalaLearner(
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_interval_updates=50,
        logs_dir=run_dir / "logs",
        logging_interval_updates=10,
    )

    for _ in range(100):
        learner.update(_example_batch())

    log_path = run_dir / "logs" / "training_metrics.jsonl"
    is_valid, message = TrainingLogger.validate_jsonl(log_path)
    print(f"validation: {is_valid} ({message})")

    records = TrainingLogger.read_jsonl(log_path)
    print(f"records: {len(records)}")
    print(f"last rho p90: {records[-1]['vtrace_rho_p90']:.3f}")
    print(f"last entropy: {records[-1]['entropy']:.3f}")


def example_read_logs() -> None:
    log_path = Path("runs/example_run/logs/training_metrics.jsonl")
    records = TrainingLogger.read_jsonl(log_path)
    latest = records[-1]
    print(
        "latest record:",
        {
            "update_count": latest["update_count"],
            "policy_version": latest["policy_version"],
            "throughput_updates_per_sec": latest["throughput_updates_per_sec"],
            "vtrace_clip_rate": latest["vtrace_clip_rate"],
        },
    )


def print_scope_note() -> None:
    print("train.py wiring is intentionally not shown here; that belongs to M3-08 integration work.")


if __name__ == "__main__":
    example_training_loop()
    example_read_logs()
    print_scope_note()
