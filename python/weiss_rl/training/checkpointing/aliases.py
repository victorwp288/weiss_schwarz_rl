"""Checkpoint alias tracker and publication helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.training.checkpointing.alias_candidates import (
    CheckpointAliasCandidate,
    checkpoint_alias_candidate,
    dev_eval_candidate_diagnostics,
    should_update_observed_best,
)
from weiss_rl.training.checkpointing.alias_mutation import (
    build_checkpoint_record,
    relative_path_text,
)
from weiss_rl.training.checkpointing.alias_publication import (
    CheckpointAliasPublication,
    apply_checkpoint_alias_publication,
    best_checkpoint_alias_mutation,
    latest_checkpoint_alias_mutation,
    maybe_publish_best_checkpoint_alias,
    maybe_publish_observed_best_checkpoint_alias,
    observed_best_checkpoint_alias_mutation,
    observed_best_checkpoint_path,
    publish_latest_checkpoint_alias,
    tracker_record,
)
from weiss_rl.training.checkpointing.tracker import (
    CHECKPOINT_TRACKER_FILENAME,
    CHECKPOINT_TRACKER_FORMAT,
    CheckpointTrainingPaths,
    best_checkpoint_record,
    default_checkpoint_tracker_payload,
    load_checkpoint_tracker,
    write_checkpoint_tracker,
)


class CheckpointAliasPaths(CheckpointTrainingPaths, Protocol):
    @property
    def latest_checkpoint_path(self) -> Path: ...

    @property
    def best_checkpoint_path(self) -> Path: ...


class LearnerRecordSource(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


def publish_checkpoint_aliases(
    *,
    stack: Any,
    training_paths: CheckpointAliasPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tracker = load_checkpoint_tracker(training_paths)
    candidate = checkpoint_alias_candidate(
        stack=stack,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    publication = apply_checkpoint_alias_publication(
        tracker=tracker,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        candidate=candidate,
    )

    write_checkpoint_tracker(training_paths, publication.tracker)
    return publication.tracker


__all__ = [
    "CHECKPOINT_TRACKER_FILENAME",
    "CHECKPOINT_TRACKER_FORMAT",
    "CheckpointAliasCandidate",
    "CheckpointAliasPublication",
    "CheckpointAliasPaths",
    "CheckpointTrainingPaths",
    "LearnerRecordSource",
    "apply_checkpoint_alias_publication",
    "best_checkpoint_record",
    "best_checkpoint_alias_mutation",
    "build_checkpoint_record",
    "checkpoint_alias_candidate",
    "default_checkpoint_tracker_payload",
    "dev_eval_candidate_diagnostics",
    "load_checkpoint_tracker",
    "maybe_publish_best_checkpoint_alias",
    "maybe_publish_observed_best_checkpoint_alias",
    "observed_best_checkpoint_path",
    "latest_checkpoint_alias_mutation",
    "observed_best_checkpoint_alias_mutation",
    "publish_checkpoint_aliases",
    "publish_latest_checkpoint_alias",
    "relative_path_text",
    "should_update_observed_best",
    "tracker_record",
    "write_checkpoint_tracker",
]
