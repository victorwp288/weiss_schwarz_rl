from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
    SELECTED_CANDIDATE_POLICY_ID,
)


def _run_relative_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_snapshot_checkpoint_path(*, repo_root: Path, run_dir: Path, policy_id: str) -> Path:
    resolved_run_dir = _run_relative_path(repo_root, run_dir)
    registry_path = resolved_run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise SystemExit(f"--init-from-run-dir snapshot registry not found: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(snapshots, list):
        raise SystemExit(f"snapshot registry must contain a snapshots list: {registry_path}")
    normalized_policy_id = str(policy_id).strip()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if str(snapshot.get("policy_id", "")).strip() != normalized_policy_id:
            continue
        update = snapshot.get("update", snapshot.get("update_count"))
        if not isinstance(update, int):
            raise SystemExit(f"snapshot {normalized_policy_id!r} is missing an integer update in {registry_path}")
        checkpoint_path = resolved_run_dir / "training" / "checkpoints" / f"checkpoint_{int(update)}.pt"
        if not checkpoint_path.is_file():
            raise SystemExit(
                f"checkpoint for snapshot {normalized_policy_id!r} was not found: {checkpoint_path}. "
                "Use --init-from-checkpoint if the source checkpoint was moved."
            )
        return checkpoint_path
    raise SystemExit(f"snapshot policy id not found in {registry_path}: {normalized_policy_id}")


def _resolve_b1_seed_checkpoint_path(
    *,
    repo_root: Path,
    run_dir: Path,
    init_policy_id: str,
) -> tuple[Path, str]:
    requested_policy_id = str(init_policy_id).strip()
    policy_ids = (
        (SELECTED_CANDIDATE_POLICY_ID, NOLEAGUE_BASELINE_POLICY_ID, NOLEAGUE_BASELINE_NAME)
        if requested_policy_id in {"", "auto"}
        else (requested_policy_id,)
    )
    failures: list[str] = []
    for policy_id in policy_ids:
        try:
            return (
                _resolve_snapshot_checkpoint_path(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    policy_id=policy_id,
                ),
                policy_id,
            )
        except SystemExit as exc:
            failures.append(str(exc))
    raise SystemExit(
        "Could not resolve a B1 seed checkpoint from --b1-run. "
        "Tried policy ids: "
        f"{', '.join(policy_ids)}. "
        f"Last error: {failures[-1] if failures else 'none'}"
    )
