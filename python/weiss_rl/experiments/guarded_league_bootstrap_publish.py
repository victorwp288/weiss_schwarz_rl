from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.experiments.bootstrap_commands import repo_relative, resolve_snapshot_checkpoint_path


def selected_snapshot_policy_id(*, selected: Mapping[str, Any], selection_json: Path) -> str:
    policy_id = str(selected.get("snapshot_policy_id", "")).strip()
    if not policy_id:
        raise RuntimeError(f"candidate selector did not record a snapshot_policy_id: {selection_json}")
    return policy_id


def resolve_selected_snapshot_checkpoint(
    *,
    selected: Mapping[str, Any],
    selection_json: Path,
    run_dir: Path,
) -> Path:
    return resolve_snapshot_checkpoint_path(
        run_dir=run_dir,
        policy_id=selected_snapshot_policy_id(selected=selected, selection_json=selection_json),
    )


def record_selected_checkpoint(
    *,
    segment_record: MutableMapping[str, Any],
    selected_checkpoint: Path,
    repo_root: Path,
) -> None:
    segment_record["selected_checkpoint"] = repo_relative(selected_checkpoint, repo_root=repo_root).as_posix()


def populate_published_segment_record(
    *,
    segment_record: MutableMapping[str, Any],
    published_selected: Mapping[str, Any],
    selected_alias_policy_id: str,
    selected_checkpoint: Path,
    repo_root: Path,
) -> None:
    segment_record["published_selected"] = dict(published_selected)
    segment_record["selected_alias_policy_id"] = str(selected_alias_policy_id)
    record_selected_checkpoint(
        segment_record=segment_record,
        selected_checkpoint=selected_checkpoint,
        repo_root=repo_root,
    )
    segment_record["status"] = "accepted"


__all__ = [
    "populate_published_segment_record",
    "record_selected_checkpoint",
    "resolve_selected_snapshot_checkpoint",
    "selected_snapshot_policy_id",
]
