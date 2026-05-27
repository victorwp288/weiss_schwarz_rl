from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry


def augment_eval_snapshot_registry(
    *,
    target_run_dir: Path,
    source_registry_json: Path,
    output_registry_json: Path | None = None,
    include_policy_ids: Sequence[str] = (),
    include_source_champions: bool = False,
    mark_imported_champions: bool = True,
) -> dict[str, Any]:
    target_run = Path(target_run_dir).resolve()
    source_registry_path = Path(source_registry_json).resolve()
    source_run = _registry_run_dir(source_registry_path)
    output_path = (
        Path(output_registry_json).resolve()
        if output_registry_json is not None
        else target_run / "training" / "snapshots" / "registry_with_imported_champions.json"
    )
    target_registry_path = target_run / "training" / "snapshots" / REGISTRY_FILENAME
    target_registry = SnapshotRegistry.load(target_registry_path)
    source_registry = SnapshotRegistry.load(source_registry_path)
    source_by_policy_id = {snapshot.policy_id: snapshot for snapshot in source_registry.snapshots}
    requested_policy_ids = _unique_policy_ids(
        [
            *include_policy_ids,
            *(source_registry.champion_snapshots if include_source_champions else ()),
        ]
    )
    copied_policy_ids: list[str] = []
    already_present_policy_ids: list[str] = []
    missing_policy_ids: list[str] = []
    target_registry.champion_size = max(int(target_registry.champion_size), int(source_registry.champion_size))
    for policy_id in requested_policy_ids:
        snapshot = source_by_policy_id.get(policy_id)
        if snapshot is None:
            missing_policy_ids.append(policy_id)
            continue
        source_weights_path = source_run / snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"source snapshot weights not found: {source_weights_path}")
        target_snapshot_dir = target_run / "training" / "snapshots" / policy_id
        source_snapshot_dir = source_weights_path.parent
        if target_snapshot_dir.exists():
            already_present_policy_ids.append(policy_id)
        else:
            shutil.copytree(source_snapshot_dir, target_snapshot_dir)
            copied_policy_ids.append(policy_id)
        target_registry.add_snapshot(
            policy_id=snapshot.policy_id,
            update=int(snapshot.update),
            weights_sha256=snapshot.weights_sha256,
            path=snapshot.path,
            created_utc=snapshot.created_utc,
        )
        if mark_imported_champions and policy_id in source_registry.champion_snapshots:
            target_registry.add_champion(policy_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_registry.save(output_path)
    summary = {
        "kind": "eval_snapshot_registry_augmentation_v1",
        "target_run_dir": target_run.as_posix(),
        "source_registry_json": source_registry_path.as_posix(),
        "source_run_dir": source_run.as_posix(),
        "target_registry_json": target_registry_path.as_posix(),
        "output_registry_json": output_path.as_posix(),
        "include_policy_ids": list(include_policy_ids),
        "include_source_champions": bool(include_source_champions),
        "mark_imported_champions": bool(mark_imported_champions),
        "copied_policy_ids": copied_policy_ids,
        "already_present_policy_ids": already_present_policy_ids,
        "missing_policy_ids": missing_policy_ids,
        "champion_snapshots": list(target_registry.champion_snapshots),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = summary_path.as_posix()
    return summary


def _registry_run_dir(registry_path: Path) -> Path:
    resolved = Path(registry_path).resolve()
    if resolved.parts[-3:] != ("training", "snapshots", REGISTRY_FILENAME):
        raise ValueError(
            "source registry path must be the canonical "
            "<run>/training/snapshots/registry.json path so snapshot weights resolve unambiguously"
        )
    return resolved.parent.parent.parent


def _unique_policy_ids(policy_ids: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for policy_id in policy_ids:
        normalized = str(policy_id).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return tuple(unique)
