"""Profiled runtime collection and learner update helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class RuntimeTrainingBatchResult(Protocol):
    learner_batch: Any
    runtime_metrics: Mapping[str, float]


def collect_runtime_training_batch(
    *,
    runtime: Any,
    algorithm: Any,
    training_config: Any,
    rewards_config: Any,
    profile_timers: bool,
    actor_torch_threads: int | None,
    collect_training_batch: Any,
    profile_block: Any,
    torch_num_threads_scope: Any,
) -> RuntimeTrainingBatchResult:
    with (
        profile_block(profile_timers, "collect_update_batch"),
        torch_num_threads_scope(actor_torch_threads),
    ):
        return collect_training_batch(
            runtime=runtime,
            algorithm=algorithm,
            training_config=training_config,
            rewards_config=rewards_config,
        )


def apply_learner_training_batch(
    *,
    learner: Any,
    learner_batch: Any,
    profile_timers: bool,
    learner_torch_threads: int | None,
    profile_block: Any,
    torch_num_threads_scope: Any,
) -> dict[str, float]:
    with profile_block(profile_timers, "learner_update"), torch_num_threads_scope(learner_torch_threads):
        return learner.update(learner_batch)


__all__ = [
    "RuntimeTrainingBatchResult",
    "apply_learner_training_batch",
    "collect_runtime_training_batch",
]
