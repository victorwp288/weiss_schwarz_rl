"""Checkpoint guard lifecycle helpers for training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.training.checkpointing.aliases import (
    CheckpointAliasPaths,
    LearnerRecordSource,
)
from weiss_rl.training.checkpointing.lifecycle_decisions import (
    FinalizeToBestDecision,
    RollbackToBestDecision,
    finalize_to_best_decision,
    finalize_to_best_event_payload,
    rollback_to_best_decision,
    rollback_to_best_event_payload,
)
from weiss_rl.training.checkpointing.lifecycle_plans import (
    finalize_lifecycle_decision,
    rollback_lifecycle_decision,
)
from weiss_rl.training.checkpointing.lifecycle_transitions import (
    apply_finalize_decision_to_event_payload,
    apply_rollback_decision_to_event_payload,
)
from weiss_rl.training.checkpointing.structured_guard import (
    extract_structured_guard_b2_anchor_score,
    structured_mainmove_guard_warning_payload,
)


class CheckpointGuardPaths(CheckpointAliasPaths, Protocol):
    @property
    def logs_dir(self) -> Path: ...

    @property
    def snapshots_dir(self) -> Path: ...


class CheckpointGuardRuntime(Protocol):
    def maybe_publish_snapshot(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def reset_outcome_tracker(self) -> None: ...

    def refresh_opponent_pool(self) -> None: ...


def checkpoint_guard_log_path(training_paths: CheckpointGuardPaths) -> Path:
    return training_paths.logs_dir / "checkpoint_guard.jsonl"


def append_checkpoint_guard_event(training_paths: CheckpointGuardPaths, payload: Mapping[str, Any]) -> None:
    path = checkpoint_guard_log_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def maybe_log_structured_mainmove_guard(
    *,
    training_paths: CheckpointGuardPaths,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    payload = structured_mainmove_guard_warning_payload(
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    if payload is None:
        return None
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def maybe_rollback_to_best_checkpoint(
    *,
    stack: Any,
    training_paths: CheckpointGuardPaths,
    run_dir: Path,
    runtime: CheckpointGuardRuntime,
    learner: LearnerRecordSource,
    learner_model: Any,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
    restore_checkpoint: Any,
    write_checkpoint: Any,
) -> dict[str, Any] | None:
    decision = rollback_lifecycle_decision(
        stack=stack,
        training_paths=training_paths,
        learner_update_count=int(learner.update_count),
        dev_eval_summary=dev_eval_summary,
        last_rollback_update=last_rollback_update,
    )
    if decision is None:
        return None

    payload = apply_rollback_decision_to_event_payload(
        training_paths=training_paths,
        run_dir=run_dir,
        runtime=runtime,
        learner=learner,
        learner_model=learner_model,
        latest_metrics=latest_metrics,
        decision=decision,
        restore_checkpoint=restore_checkpoint,
    )
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def maybe_finalize_from_best_checkpoint(
    *,
    stack: Any,
    training_paths: CheckpointGuardPaths,
    run_dir: Path,
    runtime: CheckpointGuardRuntime,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    restore_checkpoint: Any,
    ensure_current_checkpoint: Any,
) -> dict[str, Any] | None:
    decision = finalize_lifecycle_decision(
        stack=stack,
        training_paths=training_paths,
        dev_eval_summary=dev_eval_summary,
    )
    if decision is None:
        return None
    payload = apply_finalize_decision_to_event_payload(
        training_paths=training_paths,
        run_dir=run_dir,
        runtime=runtime,
        learner=learner,
        latest_metrics=latest_metrics,
        decision=decision,
        restore_checkpoint=restore_checkpoint,
    )
    append_checkpoint_guard_event(training_paths, payload)
    return payload


__all__ = [
    "CheckpointGuardPaths",
    "CheckpointGuardRuntime",
    "FinalizeToBestDecision",
    "RollbackToBestDecision",
    "append_checkpoint_guard_event",
    "checkpoint_guard_log_path",
    "extract_structured_guard_b2_anchor_score",
    "finalize_lifecycle_decision",
    "finalize_to_best_decision",
    "finalize_to_best_event_payload",
    "maybe_finalize_from_best_checkpoint",
    "maybe_log_structured_mainmove_guard",
    "maybe_rollback_to_best_checkpoint",
    "rollback_lifecycle_decision",
    "rollback_to_best_decision",
    "rollback_to_best_event_payload",
]
