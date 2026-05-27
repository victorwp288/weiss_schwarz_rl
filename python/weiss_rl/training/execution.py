"""Training execution setup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.training.inputs import require_positive_int


@dataclass(frozen=True, slots=True)
class TrainingExecutionSettings:
    checkpoint_interval_updates: int
    profile_timers: bool
    torch_profiler: bool
    b1_baseline_run_dir: Path | None = None
    seed_snapshot_run_dir: Path | None = None
    init_from_checkpoint_path: Path | None = None


def resolve_training_execution_settings(
    *,
    training_config: Any,
    checkpoint_interval_override: int | None,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    init_from_checkpoint: Path | None = None,
) -> TrainingExecutionSettings:
    """Resolve CLI/config execution controls for the minimal training runner."""

    checkpoint_interval_updates = require_positive_int(
        "--checkpoint-interval-updates",
        checkpoint_interval_override
        if checkpoint_interval_override is not None
        else int(training_config.checkpoint_interval_updates),
    )
    return TrainingExecutionSettings(
        checkpoint_interval_updates=checkpoint_interval_updates,
        profile_timers=bool(training_config.profile_timers),
        torch_profiler=bool(training_config.torch_profiler),
        b1_baseline_run_dir=None if b1_baseline_run_dir is None else b1_baseline_run_dir.resolve(),
        seed_snapshot_run_dir=None if seed_snapshot_run_dir is None else seed_snapshot_run_dir.resolve(),
        init_from_checkpoint_path=None if init_from_checkpoint is None else init_from_checkpoint.resolve(),
    )
