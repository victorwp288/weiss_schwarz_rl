from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotMeta, SnapshotRegistry

NOLEAGUE_BASELINE_ROLE = "baseline_noleague"
LEGACY_NOLEAGUE_BASELINE_MODE = "b1_no_league"
NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
NOLEAGUE_BASELINE_POLICY_ID_CANDIDATES = (
    NOLEAGUE_BASELINE_POLICY_ID,
    NOLEAGUE_BASELINE_NAME,
)
SELECTED_CANDIDATE_POLICY_ID = "selected_candidate"
SELECTED_CANDIDATE_METADATA_FORMAT = "selected_candidate_alias_metadata_v1"


def is_noleague_baseline_role(role: str) -> bool:
    return str(role).strip() == NOLEAGUE_BASELINE_ROLE


def canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    config = config_canonical.get("config")
    return config if isinstance(config, Mapping) else config_canonical


def role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    experiment = canonical_config_sections(config_canonical).get("experiment", {})
    if isinstance(experiment, Mapping):
        role = str(experiment.get("role", "")).strip()
        if role:
            return role
    return ""


def legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    training_family = canonical_config_sections(config_canonical).get("training_family_a", {})
    if isinstance(training_family, Mapping):
        return str(training_family.get("mode", "")).strip()
    return ""


def config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    role = role_from_config_canonical(config_canonical)
    if role:
        return is_noleague_baseline_role(role)
    legacy_mode = legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode:
        return legacy_mode == LEGACY_NOLEAGUE_BASELINE_MODE
    return False


def find_noleague_baseline_snapshot(
    run_dir: Path,
    *,
    policy_id_candidates: Sequence[str] = NOLEAGUE_BASELINE_POLICY_ID_CANDIDATES,
    allow_selected_candidate: bool = True,
) -> SnapshotMeta | None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    for policy_id in policy_id_candidates:
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is not None:
            return snapshot

    if allow_selected_candidate:
        selected_snapshot = snapshots_by_id.get(SELECTED_CANDIDATE_POLICY_ID)
        if selected_snapshot is not None and selected_candidate_is_locked_b1(run_dir, selected_snapshot):
            return selected_snapshot

    return None


def selected_candidate_is_locked_b1(run_dir: Path, snapshot: SnapshotMeta | None = None) -> bool:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return False
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {entry.policy_id: entry for entry in registry.snapshots}
    selected_snapshot = snapshot or snapshots_by_id.get(SELECTED_CANDIDATE_POLICY_ID)
    if selected_snapshot is None:
        return False
    if selected_snapshot.policy_id != SELECTED_CANDIDATE_POLICY_ID:
        return False
    if SELECTED_CANDIDATE_POLICY_ID not in set(registry.pinned_snapshots):
        return False

    metadata_path = layout.training_snapshots_dir / SELECTED_CANDIDATE_POLICY_ID / "policy_meta.json"
    metadata = _read_json_dict(metadata_path)
    if metadata.get("format") != SELECTED_CANDIDATE_METADATA_FORMAT:
        return False
    if metadata.get("policy_id") != SELECTED_CANDIDATE_POLICY_ID:
        return False
    if int(metadata.get("update", -1)) != int(selected_snapshot.update):
        return False
    if str(metadata.get("weights_sha256", "")).strip() != str(selected_snapshot.weights_sha256).strip():
        return False

    readiness = _read_json_dict(layout.run_dir / "paper_readiness_summary.json")
    if readiness.get("passed") is not True:
        return False
    checks = readiness.get("checks")
    if not isinstance(checks, Mapping):
        return False
    baseline_check = checks.get("baseline_win_rate_vs_b0")
    if isinstance(baseline_check, Mapping) and baseline_check.get("passed") is False:
        return False
    return True


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
