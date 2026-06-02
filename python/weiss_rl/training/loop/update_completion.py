"""Completion metrics and logger side effects for one learner update."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES = (
    "trajectory_bc_replay_",
    "paired_swing_replay_",
    "paired_outcome_preference_replay_",
    "pfsp_",
    "collector_pfsp_",
)


@dataclass(frozen=True, slots=True)
class TrainingUpdateCompletionMetrics:
    runtime: Mapping[str, float]
    schedule: Mapping[str, float]
    snapshot: Mapping[str, float]

    def apply_to(self, latest_metrics: dict[str, float]) -> dict[str, float]:
        latest_metrics.update(self.runtime)
        latest_metrics.update(self.schedule)
        latest_metrics.update(self.snapshot)
        return latest_metrics


def publish_runtime_snapshot_after_update(
    *,
    runtime: Any,
    model: Any,
    learner: Any,
    profile_timers: bool,
    profile_block: Any,
) -> dict[str, float]:
    with profile_block(profile_timers, "runtime_snapshot_publish"):
        return runtime.maybe_publish_snapshot(
            learner_model=model,
            learner_update_count=int(learner.update_count),
        )


def collect_training_update_completion_metrics(
    *,
    learner: Any,
    model: Any,
    runtime: Any,
    runtime_metrics: Mapping[str, float],
    schedule_metrics: Mapping[str, float],
    profile_timers: bool,
    profile_block: Any,
) -> TrainingUpdateCompletionMetrics:
    return TrainingUpdateCompletionMetrics(
        runtime=runtime_metrics,
        schedule=schedule_metrics,
        snapshot=publish_runtime_snapshot_after_update(
            runtime=runtime,
            model=model,
            learner=learner,
            profile_timers=profile_timers,
            profile_block=profile_block,
        ),
    )


def merge_post_update_auxiliary_metrics_into_training_log(*, learner: Any, metrics: Mapping[str, float]) -> None:
    logger = getattr(learner, "logger", None)
    if logger is None:
        return
    merge_latest = getattr(logger, "merge_latest_custom_metrics", None)
    if not callable(merge_latest):
        return
    merge_latest(
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        metrics=metrics,
        prefixes=POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    )


def complete_training_update_metrics(
    *,
    learner: Any,
    model: Any,
    runtime: Any,
    latest_metrics: dict[str, float],
    runtime_metrics: Mapping[str, float],
    schedule_metrics: Mapping[str, float],
    profile_timers: bool,
    profile_block: Any,
) -> dict[str, float]:
    completion_metrics = collect_training_update_completion_metrics(
        learner=learner,
        model=model,
        runtime=runtime,
        runtime_metrics=runtime_metrics,
        schedule_metrics=schedule_metrics,
        profile_timers=profile_timers,
        profile_block=profile_block,
    )
    completion_metrics.apply_to(latest_metrics)
    merge_post_update_auxiliary_metrics_into_training_log(learner=learner, metrics=latest_metrics)
    return latest_metrics


__all__ = [
    "POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES",
    "TrainingUpdateCompletionMetrics",
    "collect_training_update_completion_metrics",
    "complete_training_update_metrics",
    "merge_post_update_auxiliary_metrics_into_training_log",
    "publish_runtime_snapshot_after_update",
]
