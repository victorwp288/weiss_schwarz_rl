"""Post-update checkpoint and periodic dev-eval orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.periodic_dev_eval import PeriodicDevEvalGuardResult


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


def run_post_update_checkpoint_and_dev_eval_from_context(
    *,
    progress: Any,
    context: PostUpdateCheckpointDevEvalContext,
    schedule: PostUpdateCheckpointDevEvalSchedule,
    hooks: PostUpdateCheckpointDevEvalHooks,
) -> bool:
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
    "FinalTrainingCheckpointContext",
    "FinalTrainingCheckpointHooks",
    "PostUpdateCheckpointDevEvalContext",
    "PostUpdateCheckpointDevEvalHooks",
    "PostUpdateCheckpointDevEvalSchedule",
    "finalize_training_loop_progress_from_context",
    "run_post_update_checkpoint_and_dev_eval_from_context",
]
