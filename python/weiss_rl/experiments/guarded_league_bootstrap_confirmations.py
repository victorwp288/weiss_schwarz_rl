from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from weiss_rl.experiments.bootstrap_commands import repo_relative
from weiss_rl.experiments.guarded_league_bootstrap_segments import targeted_confirm_command_payloads
from weiss_rl.experiments.guarded_league_bootstrap_selection import (
    load_targeted_confirm_scores,
    targeted_confirm_summary_path,
)


class SnapshotCandidateLike(Protocol):
    @property
    def policy_id(self) -> str: ...

    @property
    def update(self) -> int: ...

    @property
    def checkpoint_path(self) -> Path: ...


def confirm_focal_policy_ids(
    *,
    preselected: Mapping[str, Any] | None,
    recent_candidates: Sequence[SnapshotCandidateLike],
    latest_policy_id: str,
    limit: int,
) -> list[str]:
    focal_policy_ids: list[str] = []
    preselected_policy_id = str(preselected.get("snapshot_policy_id", "")).strip() if preselected is not None else ""
    if preselected_policy_id:
        focal_policy_ids.append(preselected_policy_id)
    for candidate in recent_candidates:
        if len(focal_policy_ids) >= int(limit):
            break
        if candidate.policy_id not in focal_policy_ids:
            focal_policy_ids.append(candidate.policy_id)
    if not focal_policy_ids:
        focal_policy_ids.append(str(latest_policy_id))
    return focal_policy_ids


def populate_confirm_candidate_segment_record(
    *,
    segment_record: MutableMapping[str, Any],
    latest: SnapshotCandidateLike,
    repo_root: Path,
    confirm_recent_candidate_count: int,
    focal_policy_ids: Sequence[str],
    preselected: Mapping[str, Any] | None,
) -> None:
    segment_record["latest_policy_id"] = latest.policy_id
    segment_record["latest_update"] = int(latest.update)
    segment_record["latest_checkpoint"] = repo_relative(latest.checkpoint_path, repo_root=repo_root).as_posix()
    segment_record["confirm_recent_candidate_count"] = int(confirm_recent_candidate_count)
    segment_record["confirm_focal_policy_ids"] = list(focal_policy_ids)
    if preselected is not None:
        segment_record["preselected"] = dict(preselected)


def record_targeted_confirm_result(
    *,
    confirm_record: MutableMapping[str, Any],
    run_dir: Path,
    paired_seeds: int,
    repo_root: Path,
) -> Path:
    confirm_summary_path = targeted_confirm_summary_path(
        run_dir=run_dir,
        output_subdir=str(confirm_record["output_subdir"]),
        paired_seeds=paired_seeds,
    )
    confirm_record["summary_path"] = repo_relative(confirm_summary_path, repo_root=repo_root).as_posix()
    confirm_record["anchor_scores"] = load_targeted_confirm_scores(confirm_summary_path)
    return confirm_summary_path


def populate_targeted_confirm_segment_record(
    *,
    segment_record: MutableMapping[str, Any],
    targeted_confirm_records: Sequence[Mapping[str, Any]],
) -> None:
    segment_record["targeted_confirm_commands"] = targeted_confirm_command_payloads(targeted_confirm_records)
    segment_record["targeted_confirm_records"] = [dict(record) for record in targeted_confirm_records]
    if targeted_confirm_records:
        first_confirm = targeted_confirm_records[0]
        segment_record["confirm_focal_policy_id"] = first_confirm["focal_policy_id"]
        segment_record["targeted_confirm_command"] = first_confirm["command"]


__all__ = [
    "SnapshotCandidateLike",
    "confirm_focal_policy_ids",
    "populate_confirm_candidate_segment_record",
    "populate_targeted_confirm_segment_record",
    "record_targeted_confirm_result",
]
