"""Learner update and auxiliary replay phase for minimal training."""

from __future__ import annotations

from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.training.loop.update_phases import (
    POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
    RuntimeTrainingBatchResult,
    TrainingUpdateCompletionMetrics,
    TrainingUpdateScheduleResult,
    apply_learner_training_batch,
    apply_training_update_schedule,
    collect_runtime_training_batch,
    collect_training_update_completion_metrics,
    complete_training_update_metrics,
    merge_post_update_auxiliary_metrics_into_training_log,
    publish_runtime_snapshot_after_update,
    schedule_update_count_for_next_update,
)
from weiss_rl.training.loop.update_step import (
    TrainingUpdateStepHooks,
    TrainingUpdateStepInputs,
    TrainingUpdateStepOptions,
    run_training_update_step_from_context,
)
from weiss_rl.training.replay_data.training_replay_dispatch import (
    TrainingReplayStates,
    reset_policy_anchor_for_fresh_preference_replay,
    reset_policy_anchor_to_current_model,
    run_post_update_replay,
    training_replay_states_from_config,
)


def run_training_update_step(
    *,
    learner: Any,
    model: Any,
    stack: StackConfig,
    runtime: Any,
    algorithm: Any,
    training_config: Any,
    rewards_config: Any,
    replay_states: TrainingReplayStates,
    device: torch.device,
    init_schedule_offset_updates: int,
    profile_timers: bool,
    actor_torch_threads: int | None,
    learner_torch_threads: int | None,
    apply_guidance_schedule_for_next_update: Any,
    entropy_coef_for_next_update: Any,
    collect_training_batch: Any,
    profile_block: Any,
    torch_num_threads_scope: Any,
) -> dict[str, float]:
    return run_training_update_step_from_context(
        inputs=TrainingUpdateStepInputs(
            learner=learner,
            model=model,
            stack=stack,
            runtime=runtime,
            algorithm=algorithm,
            training_config=training_config,
            rewards_config=rewards_config,
            replay_states=replay_states,
            device=device,
            init_schedule_offset_updates=init_schedule_offset_updates,
        ),
        options=TrainingUpdateStepOptions(
            profile_timers=bool(profile_timers),
            actor_torch_threads=actor_torch_threads,
            learner_torch_threads=learner_torch_threads,
        ),
        hooks=TrainingUpdateStepHooks(
            apply_guidance_schedule_for_next_update=apply_guidance_schedule_for_next_update,
            entropy_coef_for_next_update=entropy_coef_for_next_update,
            collect_training_batch=collect_training_batch,
            profile_block=profile_block,
            torch_num_threads_scope=torch_num_threads_scope,
        ),
    )


__all__ = [
    "POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES",
    "RuntimeTrainingBatchResult",
    "TrainingReplayStates",
    "TrainingUpdateCompletionMetrics",
    "TrainingUpdateScheduleResult",
    "TrainingUpdateStepHooks",
    "TrainingUpdateStepInputs",
    "TrainingUpdateStepOptions",
    "apply_learner_training_batch",
    "apply_training_update_schedule",
    "collect_training_update_completion_metrics",
    "collect_runtime_training_batch",
    "complete_training_update_metrics",
    "merge_post_update_auxiliary_metrics_into_training_log",
    "publish_runtime_snapshot_after_update",
    "reset_policy_anchor_for_fresh_preference_replay",
    "reset_policy_anchor_to_current_model",
    "run_post_update_replay",
    "run_training_update_step",
    "run_training_update_step_from_context",
    "schedule_update_count_for_next_update",
    "training_replay_states_from_config",
]
