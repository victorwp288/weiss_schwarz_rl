"""Checkpoint guard lifecycle gating before effects are applied."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from weiss_rl.training.checkpointing.aliases import best_checkpoint_record, load_checkpoint_tracker
from weiss_rl.training.checkpointing.guard import dev_eval_aggregate_score
from weiss_rl.training.checkpointing.lifecycle_decisions import (
    FinalizeToBestDecision,
    RollbackToBestDecision,
    finalize_to_best_decision,
    rollback_to_best_decision,
)


def rollback_lifecycle_decision(
    *,
    stack: Any,
    training_paths: Any,
    learner_update_count: int,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
) -> RollbackToBestDecision | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if last_rollback_update is not None and (int(learner_update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    if dev_eval_aggregate_score(dev_eval_summary) is None:
        return None
    tracker = load_checkpoint_tracker(training_paths)
    return rollback_to_best_decision(
        checkpoint_guard=checkpoint_guard,
        best_record=tracker.get("best"),
        learner_update_count=int(learner_update_count),
        dev_eval_summary=dev_eval_summary,
    )


def finalize_lifecycle_decision(
    *,
    stack: Any,
    training_paths: Any,
    dev_eval_summary: Mapping[str, Any] | None,
) -> FinalizeToBestDecision | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.checkpoint_guard.enabled:
        return None
    return finalize_to_best_decision(
        best_record=best_checkpoint_record(training_paths),
        dev_eval_summary=dev_eval_summary,
    )


__all__ = [
    "finalize_lifecycle_decision",
    "rollback_lifecycle_decision",
]
