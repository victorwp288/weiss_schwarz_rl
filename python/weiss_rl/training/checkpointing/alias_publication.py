"""Checkpoint alias publication decisions and side effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.training.checkpointing.alias_candidates import (
    CheckpointAliasCandidate,
    should_update_observed_best,
)
from weiss_rl.training.checkpointing.alias_mutation import (
    CheckpointAliasMutation,
    apply_checkpoint_alias_mutation,
)
from weiss_rl.training.checkpointing.guard import should_promote_best_checkpoint
from weiss_rl.training.checkpointing.resolution import OBSERVED_BEST_CHECKPOINT_FILENAME


class CheckpointAliasPublicationPaths(Protocol):
    @property
    def checkpoint_tracker_path(self) -> Path: ...

    @property
    def latest_checkpoint_path(self) -> Path: ...

    @property
    def best_checkpoint_path(self) -> Path: ...


class LearnerRecordSource(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CheckpointAliasPublication:
    tracker: dict[str, Any]
    candidate: CheckpointAliasCandidate
    latest_record: Mapping[str, Any]
    observed_best_record: Mapping[str, Any] | None
    best_record: Mapping[str, Any] | None


def observed_best_checkpoint_path(training_paths: CheckpointAliasPublicationPaths) -> Path:
    return training_paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME


def tracker_record(tracker: Mapping[str, Any], alias_name: str) -> Mapping[str, Any] | None:
    record = tracker.get(alias_name)
    return record if isinstance(record, Mapping) else None


def latest_checkpoint_alias_mutation(
    *,
    training_paths: CheckpointAliasPublicationPaths,
    checkpoint_path: Path,
    candidate: CheckpointAliasCandidate,
) -> CheckpointAliasMutation:
    return CheckpointAliasMutation(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=checkpoint_path,
        metric_kind=candidate.metric_kind,
        metric_value=candidate.metric_value,
        include_dev_eval_candidate=True,
    )


def publish_latest_checkpoint_alias(
    *,
    tracker: dict[str, Any],
    training_paths: CheckpointAliasPublicationPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidate,
) -> Mapping[str, Any]:
    return apply_checkpoint_alias_mutation(
        tracker=tracker,
        mutation=latest_checkpoint_alias_mutation(
            training_paths=training_paths,
            checkpoint_path=checkpoint_path,
            candidate=candidate,
        ),
        run_dir=run_dir,
        learner=learner,
        candidate=candidate,
    )


def observed_best_checkpoint_alias_mutation(
    *,
    training_paths: CheckpointAliasPublicationPaths,
    checkpoint_path: Path,
    candidate: CheckpointAliasCandidate,
) -> CheckpointAliasMutation:
    assert candidate.observed_score is not None
    return CheckpointAliasMutation(
        alias_name="observed_best",
        alias_path=observed_best_checkpoint_path(training_paths),
        source_checkpoint_path=checkpoint_path,
        metric_kind="dev_eval_observed_mean",
        metric_value=float(candidate.observed_score),
        include_dev_eval_candidate=True,
    )


def maybe_publish_observed_best_checkpoint_alias(
    *,
    tracker: dict[str, Any],
    training_paths: CheckpointAliasPublicationPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidate,
) -> Mapping[str, Any] | None:
    if not should_update_observed_best(
        existing_record=tracker_record(tracker, "observed_best"),
        observed_score=candidate.observed_score,
    ):
        return None
    return apply_checkpoint_alias_mutation(
        tracker=tracker,
        mutation=observed_best_checkpoint_alias_mutation(
            training_paths=training_paths,
            checkpoint_path=checkpoint_path,
            candidate=candidate,
        ),
        run_dir=run_dir,
        learner=learner,
        candidate=candidate,
    )


def best_checkpoint_alias_mutation(
    *,
    training_paths: CheckpointAliasPublicationPaths,
    checkpoint_path: Path,
    candidate: CheckpointAliasCandidate,
) -> CheckpointAliasMutation:
    return CheckpointAliasMutation(
        alias_name="best",
        alias_path=training_paths.best_checkpoint_path,
        source_checkpoint_path=checkpoint_path,
        metric_kind=candidate.metric_kind,
        metric_value=candidate.metric_value,
    )


def maybe_publish_best_checkpoint_alias(
    *,
    tracker: dict[str, Any],
    training_paths: CheckpointAliasPublicationPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidate,
) -> Mapping[str, Any] | None:
    if not should_promote_best_checkpoint(
        existing_record=tracker_record(tracker, "best"),
        candidate_kind=candidate.metric_kind,
        candidate_value=candidate.metric_value,
    ):
        return None
    return apply_checkpoint_alias_mutation(
        tracker=tracker,
        mutation=best_checkpoint_alias_mutation(
            training_paths=training_paths,
            checkpoint_path=checkpoint_path,
            candidate=candidate,
        ),
        run_dir=run_dir,
        learner=learner,
        candidate=candidate,
    )


def apply_checkpoint_alias_publication(
    *,
    tracker: dict[str, Any],
    training_paths: CheckpointAliasPublicationPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidate,
) -> CheckpointAliasPublication:
    latest_record = publish_latest_checkpoint_alias(
        tracker=tracker,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        candidate=candidate,
    )
    observed_best_record = maybe_publish_observed_best_checkpoint_alias(
        tracker=tracker,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        candidate=candidate,
    )
    best_record = maybe_publish_best_checkpoint_alias(
        tracker=tracker,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        candidate=candidate,
    )

    return CheckpointAliasPublication(
        tracker=tracker,
        candidate=candidate,
        latest_record=latest_record,
        observed_best_record=observed_best_record,
        best_record=best_record,
    )


__all__ = [
    "CheckpointAliasPublication",
    "CheckpointAliasPublicationPaths",
    "LearnerRecordSource",
    "apply_checkpoint_alias_publication",
    "best_checkpoint_alias_mutation",
    "latest_checkpoint_alias_mutation",
    "maybe_publish_best_checkpoint_alias",
    "maybe_publish_observed_best_checkpoint_alias",
    "observed_best_checkpoint_alias_mutation",
    "observed_best_checkpoint_path",
    "publish_latest_checkpoint_alias",
    "tracker_record",
]
