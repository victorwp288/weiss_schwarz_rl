"""Checkpoint aliasing and guard application after periodic dev-eval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class CheckpointGuardApplicationResult:
    tracker_payload: Mapping[str, Any]
    next_rollback_update: int | None
    stop_requested: bool


def apply_periodic_dev_eval_checkpoint_guard(
    *,
    hooks: Any,
    stack: Any,
    learner: Any,
    model: Any,
    artifacts: Any,
    training_paths: Any,
    runtime: Any,
    device: Any,
    spec_hash256: str,
    algorithm: Any,
    latest_metrics: dict[str, float],
    effective_summary: Mapping[str, Any],
    last_checkpoint_guard_rollback_update: int | None,
    run_id256: str,
    tensorboard_logger: Any,
    update_count: int,
) -> CheckpointGuardApplicationResult:
    checkpoint_path = hooks.ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    tracker_payload = hooks.publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=effective_summary,
    )
    hooks.maybe_log_structured_mainmove_guard(
        training_paths=training_paths,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=effective_summary,
    )
    guard_event = hooks.maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        model=model,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=effective_summary,
        last_rollback_update=last_checkpoint_guard_rollback_update,
    )

    next_rollback_update = last_checkpoint_guard_rollback_update
    checkpoint_guard_stop_requested = False
    if guard_event is not None:
        next_rollback_update = update_count
        print(
            "Checkpoint guard rollback: "
            f"update={guard_event['update_count']} "
            f"best_update={guard_event['best_update_count']} "
            f"current_score={float(guard_event['current_score']):.4f} "
            f"best_score={float(guard_event['best_score']):.4f} "
            f"reasons={','.join(cast(list[str], guard_event['reasons']))}"
        )
        curriculum = stack.config.curriculum
        checkpoint_guard_stop_requested = bool(
            curriculum is not None and getattr(curriculum.checkpoint_guard, "stop_after_rollback", False)
        )
        if checkpoint_guard_stop_requested:
            latest_metrics["checkpoint_guard_stop_after_rollback"] = 1.0
    if tensorboard_logger is not None:
        tensorboard_logger.log_periodic_dev_eval(effective_summary, step=update_count)
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=update_count)
    if checkpoint_guard_stop_requested:
        if guard_event is None:
            raise RuntimeError("checkpoint guard stop was requested without a guard event")
        print(
            "Checkpoint guard early stop after rollback: "
            f"update={guard_event['update_count']} "
            f"best_update={guard_event['best_update_count']}"
        )

    return CheckpointGuardApplicationResult(
        tracker_payload=cast(Mapping[str, Any], tracker_payload),
        next_rollback_update=next_rollback_update,
        stop_requested=checkpoint_guard_stop_requested,
    )


__all__ = [
    "CheckpointGuardApplicationResult",
    "apply_periodic_dev_eval_checkpoint_guard",
]
