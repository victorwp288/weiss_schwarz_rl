"""Per-update progress state and side effects for the training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger
from weiss_rl.training.checkpointing.periodic_dev_eval import PeriodicDevEvalGuardResult
from weiss_rl.training.loop.post_update import (
    FinalTrainingCheckpointContext,
    FinalTrainingCheckpointHooks,
    PostUpdateCheckpointDevEvalContext,
    PostUpdateCheckpointDevEvalHooks,
    PostUpdateCheckpointDevEvalSchedule,
    finalize_training_loop_progress_from_context,
    run_post_update_checkpoint_and_dev_eval_from_context,
)


@dataclass(slots=True)
class TrainingLoopProgress:
    latest_metrics: dict[str, float]
    last_dev_eval_summary: Mapping[str, Any] | None = None
    last_dev_eval_update_count: int | None = None
    last_checkpoint_guard_rollback_update: int | None = None

    def record_latest_metrics(self, latest_metrics: dict[str, float]) -> None:
        self.latest_metrics = latest_metrics

    def apply_dev_eval_result(self, result: PeriodicDevEvalGuardResult) -> bool:
        self.last_dev_eval_summary = result.last_dev_eval_summary
        self.last_dev_eval_update_count = result.last_dev_eval_update_count
        self.last_checkpoint_guard_rollback_update = result.last_checkpoint_guard_rollback_update
        return bool(result.stop_requested)


def write_training_update_outputs(
    *,
    progress: TrainingLoopProgress,
    learner: Any,
    training_paths: Any,
    start_time: float,
    tensorboard_logger: TensorBoardLogger | None,
    write_scalars_record: Callable[..., Any],
) -> None:
    write_scalars_record(
        scalars_path=training_paths.scalars_path,
        learner=learner,
        metrics=progress.latest_metrics,
        start_time=start_time,
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_training_step(
            update_count=int(learner.update_count),
            policy_version=int(learner.get_policy_version()),
            wall_clock_seconds=time.time() - start_time,
            metrics=progress.latest_metrics,
        )


def run_post_update_checkpoint_and_dev_eval(
    *,
    progress: TrainingLoopProgress,
    learner: Any,
    model: Any,
    stack: Any,
    contract: Any,
    artifacts: Any,
    training_paths: Any,
    runtime: Any,
    device: Any,
    spec_hash256: str,
    algorithm: Any,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    tensorboard_logger: TensorBoardLogger | None,
    checkpoint_hooks: Any,
    periodic_dev_eval_hooks: Any,
    checkpoint_fn: Callable[..., Any],
    dev_eval_fn: Callable[..., PeriodicDevEvalGuardResult],
) -> bool:
    return run_post_update_checkpoint_and_dev_eval_from_context(
        progress=progress,
        context=PostUpdateCheckpointDevEvalContext(
            learner=learner,
            model=model,
            stack=stack,
            contract=contract,
            artifacts=artifacts,
            training_paths=training_paths,
            runtime=runtime,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
            run_id256=run_id256,
            config_hash256=config_hash256,
            tensorboard_logger=tensorboard_logger,
        ),
        schedule=PostUpdateCheckpointDevEvalSchedule(
            checkpoint_interval_updates=checkpoint_interval_updates,
        ),
        hooks=PostUpdateCheckpointDevEvalHooks(
            checkpoint_hooks=checkpoint_hooks,
            periodic_dev_eval_hooks=periodic_dev_eval_hooks,
            checkpoint_fn=checkpoint_fn,
            dev_eval_fn=dev_eval_fn,
        ),
    )


def finalize_training_loop_progress(
    *,
    progress: TrainingLoopProgress,
    learner: Any,
    stack: Any,
    artifacts: Any,
    training_paths: Any,
    runtime: Any,
    device: Any,
    spec_hash256: str,
    algorithm: Any,
    tensorboard_logger: TensorBoardLogger | None,
    hooks: Any,
    finalize_fn: Callable[..., Any],
) -> Any:
    return finalize_training_loop_progress_from_context(
        progress=progress,
        context=FinalTrainingCheckpointContext(
            learner=learner,
            stack=stack,
            artifacts=artifacts,
            training_paths=training_paths,
            runtime=runtime,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
            tensorboard_logger=tensorboard_logger,
        ),
        hooks=FinalTrainingCheckpointHooks(
            hooks=hooks,
            finalize_fn=finalize_fn,
        ),
    )


__all__ = [
    "FinalTrainingCheckpointContext",
    "FinalTrainingCheckpointHooks",
    "PostUpdateCheckpointDevEvalContext",
    "PostUpdateCheckpointDevEvalHooks",
    "PostUpdateCheckpointDevEvalSchedule",
    "TrainingLoopProgress",
    "finalize_training_loop_progress",
    "finalize_training_loop_progress_from_context",
    "run_post_update_checkpoint_and_dev_eval",
    "run_post_update_checkpoint_and_dev_eval_from_context",
    "write_training_update_outputs",
]
