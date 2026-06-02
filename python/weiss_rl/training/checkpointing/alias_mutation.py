from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CheckpointAliasMutation:
    alias_name: str
    alias_path: Path
    source_checkpoint_path: Path
    metric_kind: str | None
    metric_value: float | None
    include_dev_eval_candidate: bool = False


class LearnerRecordSource(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


class CheckpointAliasCandidateSource(Protocol):
    @property
    def dev_eval_candidate(self) -> dict[str, Any] | None: ...


def relative_path_text(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    run_dir: Path,
    learner: LearnerRecordSource,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": relative_path_text(alias_path, root=run_dir),
        "source_checkpoint_path": relative_path_text(source_checkpoint_path, root=run_dir),
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def alias_record_for_mutation(
    *,
    mutation: CheckpointAliasMutation,
    run_dir: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidateSource,
) -> dict[str, Any]:
    record = build_checkpoint_record(
        alias_name=mutation.alias_name,
        alias_path=mutation.alias_path,
        source_checkpoint_path=mutation.source_checkpoint_path,
        run_dir=run_dir,
        learner=learner,
        metric_kind=mutation.metric_kind,
        metric_value=mutation.metric_value,
    )
    if mutation.include_dev_eval_candidate and candidate.dev_eval_candidate is not None:
        record["dev_eval_candidate"] = candidate.dev_eval_candidate
    return record


def apply_checkpoint_alias_mutation(
    *,
    tracker: dict[str, Any],
    mutation: CheckpointAliasMutation,
    run_dir: Path,
    learner: LearnerRecordSource,
    candidate: CheckpointAliasCandidateSource,
) -> dict[str, Any]:
    shutil.copy2(mutation.source_checkpoint_path, mutation.alias_path)
    record = alias_record_for_mutation(
        mutation=mutation,
        run_dir=run_dir,
        learner=learner,
        candidate=candidate,
    )
    tracker[mutation.alias_name] = record
    return record


__all__ = [
    "CheckpointAliasMutation",
    "CheckpointAliasCandidateSource",
    "LearnerRecordSource",
    "alias_record_for_mutation",
    "apply_checkpoint_alias_mutation",
    "build_checkpoint_record",
    "relative_path_text",
]
