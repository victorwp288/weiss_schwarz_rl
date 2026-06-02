"""Snapshot-checkpoint resolution helpers for package workflows."""

from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
    SELECTED_CANDIDATE_POLICY_ID,
)

__all__ = [
    "resolve_b1_seed_checkpoint_path",
    "resolve_single_snapshot_checkpoint_path",
    "resolve_snapshot_checkpoint_path",
]

_B1_SEED_AUTO_POLICY_IDS = (
    NOLEAGUE_BASELINE_POLICY_ID,
    NOLEAGUE_BASELINE_NAME,
    SELECTED_CANDIDATE_POLICY_ID,
)


def _run_relative_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _snapshot_registry_path(run_dir: Path) -> Path:
    return run_dir / "training" / "snapshots" / "registry.json"


def _load_snapshot_list(registry_path: Path) -> list[object]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(snapshots, list):
        raise SystemExit(f"snapshot registry must contain a snapshots list: {registry_path}")
    return snapshots


def _snapshot_policy_ids(snapshots: list[object]) -> list[str]:
    policy_ids = [str(snapshot.get("policy_id", "")).strip() for snapshot in snapshots if isinstance(snapshot, dict)]
    return [policy_id for policy_id in policy_ids if policy_id]


def _snapshot_update(snapshot: dict[object, object], *, policy_id: str, registry_path: Path) -> int:
    update = snapshot.get("update", snapshot.get("update_count"))
    if not isinstance(update, int):
        raise SystemExit(f"snapshot {policy_id!r} is missing an integer update in {registry_path}")
    return int(update)


def _find_snapshot_update(*, snapshots: list[object], policy_id: str, registry_path: Path) -> int:
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if str(snapshot.get("policy_id", "")).strip() != policy_id:
            continue
        return _snapshot_update(snapshot, policy_id=policy_id, registry_path=registry_path)
    raise SystemExit(f"snapshot policy id not found in {registry_path}: {policy_id}")


def _b1_seed_policy_ids(init_policy_id: str) -> tuple[str, ...]:
    requested_policy_id = str(init_policy_id).strip()
    if requested_policy_id in {"", "auto"}:
        return _B1_SEED_AUTO_POLICY_IDS
    return (requested_policy_id,)


def _format_b1_seed_resolution_error(policy_ids: tuple[str, ...], failures: list[str]) -> str:
    return (
        "Could not resolve a B1 seed checkpoint from --b1-run. "
        "Tried policy ids: "
        f"{', '.join(policy_ids)}. "
        f"Last error: {failures[-1] if failures else 'none'}"
    )


def resolve_snapshot_checkpoint_path(*, repo_root: Path, run_dir: Path, policy_id: str) -> Path:
    resolved_run_dir = _run_relative_path(repo_root, run_dir)
    registry_path = _snapshot_registry_path(resolved_run_dir)
    if not registry_path.is_file():
        raise SystemExit(f"--init-from-run-dir snapshot registry not found: {registry_path}")
    snapshots = _load_snapshot_list(registry_path)
    normalized_policy_id = str(policy_id).strip()
    update = _find_snapshot_update(snapshots=snapshots, policy_id=normalized_policy_id, registry_path=registry_path)
    checkpoint_path = resolved_run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"checkpoint for snapshot {normalized_policy_id!r} was not found: {checkpoint_path}. "
            "Use --init-from-checkpoint if the source checkpoint was moved."
        )
    return checkpoint_path


def resolve_b1_seed_checkpoint_path(
    *,
    repo_root: Path,
    run_dir: Path,
    init_policy_id: str,
) -> tuple[Path, str]:
    policy_ids = _b1_seed_policy_ids(init_policy_id)
    failures: list[str] = []
    for policy_id in policy_ids:
        try:
            return (
                resolve_snapshot_checkpoint_path(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    policy_id=policy_id,
                ),
                policy_id,
            )
        except SystemExit as exc:
            failures.append(str(exc))
    raise SystemExit(_format_b1_seed_resolution_error(policy_ids, failures))


def resolve_single_snapshot_checkpoint_path(*, repo_root: Path, run_dir: Path) -> tuple[Path, str]:
    resolved_run_dir = _run_relative_path(repo_root, run_dir)
    registry_path = _snapshot_registry_path(resolved_run_dir)
    if not registry_path.is_file():
        raise SystemExit(f"smoke profile fallback requires a snapshot registry: {registry_path}")
    snapshots = _load_snapshot_list(registry_path)
    snapshot_policy_ids = _snapshot_policy_ids(snapshots)
    if len(snapshot_policy_ids) != 1:
        raise SystemExit(
            f"smoke profile fallback requires exactly one snapshot in {registry_path}; found {len(snapshot_policy_ids)}"
        )
    policy_id = snapshot_policy_ids[0]
    return (
        resolve_snapshot_checkpoint_path(
            repo_root=repo_root,
            run_dir=run_dir,
            policy_id=policy_id,
        ),
        policy_id,
    )
