from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_discovery import snapshot_by_policy_id
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)

B1_CANDIDATE_ALIAS_METADATA_FORMAT = "b1_candidate_alias_metadata_v1"
SELECTED_CANDIDATE_ALIAS_METADATA_FORMAT = "selected_candidate_alias_metadata_v1"


def publish_snapshot_alias(
    *,
    run_dir: Path,
    source_policy_id: str,
    alias_policy_id: str,
    metadata_format: str,
    selection_summary: Mapping[str, Any] | None = None,
    skip_copy_if_same_path: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    normalized_alias = str(alias_policy_id).strip()
    if not normalized_alias:
        raise ValueError("alias_policy_id must be non-empty")

    registry_path = run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")

    registry = SnapshotRegistry.load(registry_path)
    source_snapshot = snapshot_by_policy_id(run_dir).get(str(source_policy_id))
    if source_snapshot is None:
        raise ValueError(f"source policy {source_policy_id!r} is not present in {registry_path}")

    source_path = source_snapshot.get("path")
    weights_sha256 = source_snapshot.get("weights_sha256")
    update = source_snapshot.get("update")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(f"source policy {source_policy_id!r} is missing a snapshot path")
    if not isinstance(weights_sha256, str) or not weights_sha256:
        raise ValueError(f"source policy {source_policy_id!r} is missing weights_sha256")
    if isinstance(update, bool) or not isinstance(update, int):
        raise ValueError(f"source policy {source_policy_id!r} is missing integer update")

    source_weights_path = run_dir / source_path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"source weights not found: {source_weights_path}")

    target_relpath = snapshot_weights_relpath(normalized_alias)
    target_weights_path = run_dir / target_relpath
    target_weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not skip_copy_if_same_path or source_weights_path.resolve() != target_weights_path.resolve():
        shutil.copy2(source_weights_path, target_weights_path)

    metadata_path = target_weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": str(metadata_format),
        "policy_id": normalized_alias,
        "alias_for_policy_id": str(source_policy_id),
        "source_weights_path": str(source_path),
        "weights_path": target_relpath,
        "weights_sha256": weights_sha256,
        "update": int(update),
        "selection_summary": dict(selection_summary or {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry.add_snapshot(
        policy_id=normalized_alias,
        update=int(update),
        weights_sha256=weights_sha256,
        path=target_relpath,
    )
    registry.pin_snapshot(normalized_alias)
    registry.save(registry_path)
    return {
        "policy_id": normalized_alias,
        "alias_for_policy_id": str(source_policy_id),
        "update": int(update),
        "weights_path": target_relpath,
        "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
        "registry_path": registry_path.as_posix(),
    }
