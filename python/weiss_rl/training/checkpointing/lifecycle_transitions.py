"""Checkpoint guard lifecycle transitions from decisions to event payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.training.checkpointing.aliases import LearnerRecordSource, relative_path_text
from weiss_rl.training.checkpointing.lifecycle_decisions import (
    FinalizeToBestDecision,
    RollbackToBestDecision,
    finalize_to_best_event_payload,
    rollback_to_best_event_payload,
)
from weiss_rl.training.checkpointing.lifecycle_effects import (
    apply_finalize_to_best_effects,
    apply_rollback_to_best_effects,
)


def apply_rollback_decision_to_event_payload(
    *,
    training_paths: Any,
    run_dir: Path,
    runtime: Any,
    learner: LearnerRecordSource,
    learner_model: Any,
    latest_metrics: Mapping[str, float] | None,
    decision: RollbackToBestDecision,
    restore_checkpoint: Any,
) -> dict[str, Any]:
    effects = apply_rollback_to_best_effects(
        training_paths=training_paths,
        runtime=runtime,
        learner_model=learner_model,
        learner_update_count=int(learner.update_count),
        best_update_count=int(decision.best.update_count),
        restore_checkpoint=restore_checkpoint,
    )
    return rollback_to_best_event_payload(
        learner_update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        decision=decision,
        best_checkpoint_path=relative_path_text(effects.best_checkpoint_path, root=run_dir),
        latest_checkpoint_path=relative_path_text(training_paths.latest_checkpoint_path, root=run_dir),
        publish_metrics=effects.publish_metrics,
        latest_metrics=latest_metrics,
        demoted_champions=effects.demoted_champions,
    )


def apply_finalize_decision_to_event_payload(
    *,
    training_paths: Any,
    run_dir: Path,
    runtime: Any,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    decision: FinalizeToBestDecision,
    restore_checkpoint: Any,
) -> dict[str, Any]:
    effects = apply_finalize_to_best_effects(
        training_paths=training_paths,
        runtime=runtime,
        best_update_count=int(decision.best.update_count),
        restore_checkpoint=restore_checkpoint,
    )
    return finalize_to_best_event_payload(
        learner_update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        decision=decision,
        latest_metrics=latest_metrics,
        best_checkpoint_path=relative_path_text(effects.best_checkpoint_path, root=run_dir),
        latest_checkpoint_path=relative_path_text(training_paths.latest_checkpoint_path, root=run_dir),
        demoted_champions=effects.demoted_champions,
    )


__all__ = [
    "apply_finalize_decision_to_event_payload",
    "apply_rollback_decision_to_event_payload",
]
