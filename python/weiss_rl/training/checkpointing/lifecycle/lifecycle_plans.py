"""Checkpoint guard lifecycle gating before effects are applied."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.training.checkpointing.aliases.aliases import best_checkpoint_record, load_checkpoint_tracker
from weiss_rl.training.checkpointing.guards.dev_eval_metrics import dev_eval_aggregate_score
from weiss_rl.training.checkpointing.lifecycle.lifecycle_decisions import (
    FinalizeToBestDecision,
    RollbackToBestDecision,
    finalize_to_best_decision,
    rollback_to_best_decision,
)


@dataclass(frozen=True, slots=True)
class CheckpointLifecycleDecisionStep:
    step_id: str
    question: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "question": self.question,
            "evidence": list(self.evidence),
        }


CHECKPOINT_LIFECYCLE_DECISION_PLAN: tuple[CheckpointLifecycleDecisionStep, ...] = (
    CheckpointLifecycleDecisionStep(
        step_id="guard_enabled",
        question="Is checkpoint guard behavior enabled for this run?",
        evidence=("curriculum.checkpoint_guard.enabled",),
    ),
    CheckpointLifecycleDecisionStep(
        step_id="dev_eval_available",
        question="Is there a dev-eval summary with an aggregate score?",
        evidence=("dev_eval_summary.aggregate_score", "dev_eval_summary.anchors"),
    ),
    CheckpointLifecycleDecisionStep(
        step_id="cooldown_elapsed",
        question="Can rollback run again at this update?",
        evidence=("learner_update_count", "last_rollback_update", "checkpoint_guard.cooldown_updates"),
    ),
    CheckpointLifecycleDecisionStep(
        step_id="best_checkpoint_available",
        question="Is there a prior best checkpoint with a usable dev-eval score?",
        evidence=("checkpoint_tracker.best", "checkpoint_guard.min_best_score"),
    ),
    CheckpointLifecycleDecisionStep(
        step_id="quality_regression",
        question="Did current dev-eval regress enough to justify rollback or finalization to best?",
        evidence=("current aggregate score", "best aggregate score", "confidence stats", "stall/truncation rates"),
    ),
)


def checkpoint_lifecycle_decision_plan_payload() -> list[dict[str, object]]:
    return [step.as_payload() for step in CHECKPOINT_LIFECYCLE_DECISION_PLAN]


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
    "CHECKPOINT_LIFECYCLE_DECISION_PLAN",
    "CheckpointLifecycleDecisionStep",
    "checkpoint_lifecycle_decision_plan_payload",
    "finalize_lifecycle_decision",
    "rollback_lifecycle_decision",
]
