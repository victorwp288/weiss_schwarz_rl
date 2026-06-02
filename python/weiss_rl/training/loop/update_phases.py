"""Compatibility surface for one canonical learner-update phase family."""

from __future__ import annotations

from weiss_rl.training.loop.update_batch import (
    RuntimeTrainingBatchResult,
    apply_learner_training_batch,
    collect_runtime_training_batch,
)
from weiss_rl.training.loop.update_completion import (
    POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    TrainingUpdateCompletionMetrics,
    collect_training_update_completion_metrics,
    complete_training_update_metrics,
    merge_post_update_auxiliary_metrics_into_training_log,
    publish_runtime_snapshot_after_update,
)
from weiss_rl.training.loop.update_schedule import (
    TrainingUpdateScheduleResult,
    apply_training_update_schedule,
    schedule_update_count_for_next_update,
)

__all__ = [
    "POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES",
    "RuntimeTrainingBatchResult",
    "TrainingUpdateCompletionMetrics",
    "TrainingUpdateScheduleResult",
    "apply_learner_training_batch",
    "apply_training_update_schedule",
    "collect_training_update_completion_metrics",
    "collect_runtime_training_batch",
    "complete_training_update_metrics",
    "merge_post_update_auxiliary_metrics_into_training_log",
    "publish_runtime_snapshot_after_update",
    "schedule_update_count_for_next_update",
]
