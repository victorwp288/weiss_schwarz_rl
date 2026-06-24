"""Post-update checkpoint and periodic dev-eval orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.logging.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.guards.periodic_dev_eval import PeriodicDevEvalGuardResult


@dataclass(frozen=True, slots=True)
class PostUpdateStagePlanItem:
    name: str
    purpose: str


POST_UPDATE_CHECKPOINT_DEV_EVAL_PLAN: tuple[PostUpdateStagePlanItem, ...] = (
    PostUpdateStagePlanItem("checkpoint", "publish the current checkpoint and promotion aliases when scheduled"),
    PostUpdateStagePlanItem("dev_eval", "run periodic dev eval and apply checkpoint guard decisions"),
)


@dataclass(frozen=True, slots=True)
class PostUpdateCheckpointDevEvalContext:
    learner: Any
    model: Any
    stack: Any
    contract: Any
    artifacts: Any
    training_paths: Any
    runtime: Any
    device: Any
    spec_hash256: str
    algorithm: Any
    run_id256: str
    config_hash256: str
    tensorboard_logger: TensorBoardLogger | None


@dataclass(frozen=True, slots=True)
class PostUpdateCheckpointDevEvalSchedule:
    checkpoint_interval_updates: int


@dataclass(frozen=True, slots=True)
class PostUpdateCheckpointDevEvalHooks:
    checkpoint_hooks: Any
    periodic_dev_eval_hooks: Any
    checkpoint_fn: Callable[..., Any]
    dev_eval_fn: Callable[..., PeriodicDevEvalGuardResult]


def run_post_update_checkpoint_stage(
    *,
    progress: Any,
    context: PostUpdateCheckpointDevEvalContext,
    schedule: PostUpdateCheckpointDevEvalSchedule,
    hooks: PostUpdateCheckpointDevEvalHooks,
) -> None:
    hooks.checkpoint_fn(
        learner=context.learner,
        stack=context.stack,
        contract=context.contract,
        artifacts=context.artifacts,
        training_paths=context.training_paths,
        runtime=context.runtime,
        device=context.device,
        spec_hash256=context.spec_hash256,
        algorithm=context.algorithm,
        latest_metrics=progress.latest_metrics,
        last_dev_eval_summary=progress.last_dev_eval_summary,
        checkpoint_interval_updates=schedule.checkpoint_interval_updates,
        run_id256=context.run_id256,
        config_hash256=context.config_hash256,
        tensorboard_logger=context.tensorboard_logger,
        hooks=hooks.checkpoint_hooks,
    )


def run_post_update_dev_eval_stage(
    *,
    progress: Any,
    context: PostUpdateCheckpointDevEvalContext,
    hooks: PostUpdateCheckpointDevEvalHooks,
) -> bool:
    dev_eval_result = hooks.dev_eval_fn(
        learner=context.learner,
        model=context.model,
        stack=context.stack,
        contract=context.contract,
        artifacts=context.artifacts,
        training_paths=context.training_paths,
        runtime=context.runtime,
        device=context.device,
        spec_hash256=context.spec_hash256,
        algorithm=context.algorithm,
        latest_metrics=progress.latest_metrics,
        last_dev_eval_summary=progress.last_dev_eval_summary,
        last_dev_eval_update_count=progress.last_dev_eval_update_count,
        last_checkpoint_guard_rollback_update=progress.last_checkpoint_guard_rollback_update,
        run_id256=context.run_id256,
        config_hash256=context.config_hash256,
        tensorboard_logger=context.tensorboard_logger,
        hooks=hooks.periodic_dev_eval_hooks,
    )
    return progress.apply_dev_eval_result(dev_eval_result)


def run_post_update_stage(
    *,
    stage_name: str,
    progress: Any,
    context: PostUpdateCheckpointDevEvalContext,
    schedule: PostUpdateCheckpointDevEvalSchedule,
    hooks: PostUpdateCheckpointDevEvalHooks,
) -> bool:
    if stage_name == "checkpoint":
        run_post_update_checkpoint_stage(
            progress=progress,
            context=context,
            schedule=schedule,
            hooks=hooks,
        )
        return False
    if stage_name == "dev_eval":
        return run_post_update_dev_eval_stage(progress=progress, context=context, hooks=hooks)
    raise ValueError(f"unknown post-update stage: {stage_name}")


def run_post_update_checkpoint_and_dev_eval_from_context(
    *,
    progress: Any,
    context: PostUpdateCheckpointDevEvalContext,
    schedule: PostUpdateCheckpointDevEvalSchedule,
    hooks: PostUpdateCheckpointDevEvalHooks,
) -> bool:
    stop_requested = False
    for stage in POST_UPDATE_CHECKPOINT_DEV_EVAL_PLAN:
        stop_requested = run_post_update_stage(
            stage_name=stage.name,
            progress=progress,
            context=context,
            schedule=schedule,
            hooks=hooks,
        )
        if stop_requested:
            break
    return stop_requested


@dataclass(frozen=True, slots=True)
class FinalTrainingCheckpointContext:
    learner: Any
    stack: Any
    artifacts: Any
    training_paths: Any
    runtime: Any
    device: Any
    spec_hash256: str
    algorithm: Any
    tensorboard_logger: TensorBoardLogger | None


@dataclass(frozen=True, slots=True)
class FinalTrainingCheckpointHooks:
    hooks: Any
    finalize_fn: Callable[..., Any]


def finalize_training_loop_progress_from_context(
    *,
    progress: Any,
    context: FinalTrainingCheckpointContext,
    hooks: FinalTrainingCheckpointHooks,
) -> Any:
    return hooks.finalize_fn(
        learner=context.learner,
        stack=context.stack,
        artifacts=context.artifacts,
        training_paths=context.training_paths,
        runtime=context.runtime,
        device=context.device,
        spec_hash256=context.spec_hash256,
        algorithm=context.algorithm,
        latest_metrics=progress.latest_metrics,
        last_dev_eval_summary=progress.last_dev_eval_summary,
        last_dev_eval_update_count=progress.last_dev_eval_update_count,
        tensorboard_logger=context.tensorboard_logger,
        hooks=hooks.hooks,
    )


__all__ = [
    "POST_UPDATE_CHECKPOINT_DEV_EVAL_PLAN",
    "FinalTrainingCheckpointContext",
    "FinalTrainingCheckpointHooks",
    "PostUpdateStagePlanItem",
    "PostUpdateCheckpointDevEvalContext",
    "PostUpdateCheckpointDevEvalHooks",
    "PostUpdateCheckpointDevEvalSchedule",
    "finalize_training_loop_progress_from_context",
    "run_post_update_checkpoint_stage",
    "run_post_update_checkpoint_and_dev_eval_from_context",
    "run_post_update_dev_eval_stage",
    "run_post_update_stage",
]
