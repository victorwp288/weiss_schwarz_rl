"""Stage pipeline for one canonical learner update."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingUpdateStageFunctions:
    apply_training_update_schedule: Callable[..., Any]
    collect_runtime_training_batch: Callable[..., Any]
    apply_learner_training_batch: Callable[..., dict[str, float]]
    run_post_update_replay: Callable[..., None]
    complete_training_update_metrics: Callable[..., dict[str, float]]


@dataclass(frozen=True, slots=True)
class TrainingUpdateStageResults:
    schedule_result: Any
    runtime_batch: Any
    latest_metrics: dict[str, float]


def apply_training_update_schedule_stage(
    *,
    inputs: Any,
    hooks: Any,
    stage_functions: TrainingUpdateStageFunctions,
) -> Any:
    return stage_functions.apply_training_update_schedule(
        learner=inputs.learner,
        model=inputs.model,
        stack=inputs.stack,
        training_config=inputs.training_config,
        init_schedule_offset_updates=inputs.init_schedule_offset_updates,
        apply_guidance_schedule_for_next_update=hooks.apply_guidance_schedule_for_next_update,
        entropy_coef_for_next_update=hooks.entropy_coef_for_next_update,
    )


def collect_runtime_training_batch_stage(
    *,
    inputs: Any,
    options: Any,
    hooks: Any,
    stage_functions: TrainingUpdateStageFunctions,
) -> Any:
    return stage_functions.collect_runtime_training_batch(
        runtime=inputs.runtime,
        algorithm=inputs.algorithm,
        training_config=inputs.training_config,
        rewards_config=inputs.rewards_config,
        profile_timers=options.profile_timers,
        actor_torch_threads=options.actor_torch_threads,
        collect_training_batch=hooks.collect_training_batch,
        profile_block=hooks.profile_block,
        torch_num_threads_scope=hooks.torch_num_threads_scope,
    )


def apply_learner_training_batch_stage(
    *,
    inputs: Any,
    options: Any,
    hooks: Any,
    runtime_batch: Any,
    stage_functions: TrainingUpdateStageFunctions,
) -> dict[str, float]:
    return stage_functions.apply_learner_training_batch(
        learner=inputs.learner,
        learner_batch=runtime_batch.learner_batch,
        profile_timers=options.profile_timers,
        learner_torch_threads=options.learner_torch_threads,
        profile_block=hooks.profile_block,
        torch_num_threads_scope=hooks.torch_num_threads_scope,
    )


def run_post_update_replay_stage(
    *,
    inputs: Any,
    options: Any,
    hooks: Any,
    latest_metrics: dict[str, float],
    stage_functions: TrainingUpdateStageFunctions,
) -> None:
    stage_functions.run_post_update_replay(
        replay_states=inputs.replay_states,
        learner=inputs.learner,
        training_config=inputs.training_config,
        device=inputs.device,
        update_count=int(inputs.learner.update_count),
        latest_metrics=latest_metrics,
        profile_timers=bool(options.profile_timers),
        learner_torch_threads=options.learner_torch_threads,
        profile_block=hooks.profile_block,
        torch_num_threads_scope=hooks.torch_num_threads_scope,
    )


def complete_training_update_metrics_stage(
    *,
    inputs: Any,
    options: Any,
    hooks: Any,
    results: TrainingUpdateStageResults,
    stage_functions: TrainingUpdateStageFunctions,
) -> dict[str, float]:
    return stage_functions.complete_training_update_metrics(
        learner=inputs.learner,
        model=inputs.model,
        runtime=inputs.runtime,
        latest_metrics=results.latest_metrics,
        runtime_metrics=results.runtime_batch.runtime_metrics,
        schedule_metrics=results.schedule_result.metrics,
        profile_timers=options.profile_timers,
        profile_block=hooks.profile_block,
    )


def run_training_update_stage_pipeline(
    *,
    inputs: Any,
    options: Any,
    hooks: Any,
    stage_functions: TrainingUpdateStageFunctions,
) -> dict[str, float]:
    schedule_result = apply_training_update_schedule_stage(
        inputs=inputs,
        hooks=hooks,
        stage_functions=stage_functions,
    )
    runtime_batch = collect_runtime_training_batch_stage(
        inputs=inputs,
        options=options,
        hooks=hooks,
        stage_functions=stage_functions,
    )
    latest_metrics = apply_learner_training_batch_stage(
        inputs=inputs,
        options=options,
        hooks=hooks,
        runtime_batch=runtime_batch,
        stage_functions=stage_functions,
    )
    run_post_update_replay_stage(
        inputs=inputs,
        options=options,
        hooks=hooks,
        latest_metrics=latest_metrics,
        stage_functions=stage_functions,
    )
    return complete_training_update_metrics_stage(
        inputs=inputs,
        options=options,
        hooks=hooks,
        results=TrainingUpdateStageResults(
            schedule_result=schedule_result,
            runtime_batch=runtime_batch,
            latest_metrics=latest_metrics,
        ),
        stage_functions=stage_functions,
    )


__all__ = [
    "TrainingUpdateStageFunctions",
    "TrainingUpdateStageResults",
    "apply_learner_training_batch_stage",
    "apply_training_update_schedule_stage",
    "collect_runtime_training_batch_stage",
    "complete_training_update_metrics_stage",
    "run_post_update_replay_stage",
    "run_training_update_stage_pipeline",
]
