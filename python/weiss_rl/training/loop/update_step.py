"""Context-based orchestration for one learner update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.training.loop.update_phases import (
    apply_learner_training_batch,
    apply_training_update_schedule,
    collect_runtime_training_batch,
    complete_training_update_metrics,
)
from weiss_rl.training.loop.update_stage_pipeline import (
    TrainingUpdateStageFunctions,
    run_training_update_stage_pipeline,
)
from weiss_rl.training.replay_data.training_replay_dispatch import TrainingReplayStates, run_post_update_replay


@dataclass(frozen=True, slots=True)
class TrainingUpdateStepInputs:
    learner: Any
    model: Any
    stack: StackConfig
    runtime: Any
    algorithm: Any
    training_config: Any
    rewards_config: Any
    replay_states: TrainingReplayStates
    device: torch.device
    init_schedule_offset_updates: int


@dataclass(frozen=True, slots=True)
class TrainingUpdateStepOptions:
    profile_timers: bool
    actor_torch_threads: int | None
    learner_torch_threads: int | None


@dataclass(frozen=True, slots=True)
class TrainingUpdateStepHooks:
    apply_guidance_schedule_for_next_update: Any
    entropy_coef_for_next_update: Any
    collect_training_batch: Any
    profile_block: Any
    torch_num_threads_scope: Any


def run_training_update_step_from_context(
    *,
    inputs: TrainingUpdateStepInputs,
    options: TrainingUpdateStepOptions,
    hooks: TrainingUpdateStepHooks,
) -> dict[str, float]:
    return run_training_update_stage_pipeline(
        inputs=inputs,
        options=options,
        hooks=hooks,
        stage_functions=TrainingUpdateStageFunctions(
            apply_training_update_schedule=apply_training_update_schedule,
            collect_runtime_training_batch=collect_runtime_training_batch,
            apply_learner_training_batch=apply_learner_training_batch,
            run_post_update_replay=run_post_update_replay,
            complete_training_update_metrics=complete_training_update_metrics,
        ),
    )


__all__ = [
    "TrainingUpdateStepHooks",
    "TrainingUpdateStepInputs",
    "TrainingUpdateStepOptions",
    "run_training_update_step_from_context",
]
