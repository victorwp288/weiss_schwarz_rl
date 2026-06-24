"""Shared bookkeeping for IMPALA update modes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from weiss_rl.learners.update_bookkeeping import throughput_metrics


@dataclass(frozen=True, slots=True)
class ImpalaUpdateScope:
    started_at: float
    metrics: dict[str, float]


def begin_impala_update_scope(
    *,
    learner: Any,
    batch: Any,
    started_at: float | None = None,
    count_learner_update: bool,
    include_training_metrics: bool,
    checkpoint_on_interval: bool,
) -> ImpalaUpdateScope:
    update_started = time.perf_counter() if started_at is None else float(started_at)
    if count_learner_update:
        learner.update_count += 1
    batch_size = learner._batch_size(batch)
    learner.total_samples_processed += batch_size

    metrics: dict[str, float] = {}
    if include_training_metrics:
        elapsed = time.time() - learner.start_time
        throughput_samples_per_sec, throughput_updates_per_sec = throughput_metrics(
            total_samples_processed=learner.total_samples_processed,
            update_count=learner.update_count,
            elapsed_seconds=elapsed,
        )
        metrics.update(
            {
                "loss": 0.0,
                "throughput_samples_per_sec": throughput_samples_per_sec,
                "throughput_updates_per_sec": throughput_updates_per_sec,
                "entropy_coef": float(learner.entropy_coef),
            }
        )

    if (
        checkpoint_on_interval
        and learner.checkpoint_dir
        and learner.update_count % learner.checkpoint_interval_updates == 0
    ):
        learner.policy_version += 1
        learner._write_checkpoint_metadata()

    if learner.profile_timers:
        learner._active_timing_metrics = {}

    return ImpalaUpdateScope(started_at=update_started, metrics=metrics)


def set_impala_model_train_mode(learner: Any) -> None:
    learner.model.train()
    if learner.compiled_model is not None:
        learner.compiled_model.train()


def finalize_impala_update_scope(
    *,
    learner: Any,
    metrics: dict[str, float],
    started_at: float,
) -> dict[str, float]:
    learner._record_timing_ms("learner_total", time.perf_counter() - started_at)
    if learner._active_timing_metrics is not None:
        metrics.update(learner._active_timing_metrics)
        learner._active_timing_metrics = None
    return metrics


__all__ = [
    "ImpalaUpdateScope",
    "begin_impala_update_scope",
    "finalize_impala_update_scope",
    "set_impala_model_train_mode",
]
