from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

import torch

from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
    snapshot_weights_relpath,
)


class SnapshotTrainingPaths(Protocol):
    @property
    def snapshots_dir(self) -> Path: ...


class StackWithLeagueConfig(Protocol):
    @property
    def config(self) -> Any: ...


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_snapshot_artifact(
    *,
    snapshots_dir: Path,
    run_dir: Path,
    checkpoint_path: Path,
    policy_id: str,
    update: int,
    config_hash256: str,
    device: torch.device,
    model_state_dict: dict[str, Any],
    structured_policy_contract: str | None = None,
    public_heuristic_logit_bias_scale: float | None = None,
    public_heuristic_actor_logit_bias_scale: float | None = None,
) -> tuple[Path, str]:
    snapshot_dir = snapshots_dir / policy_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
    weights_payload = {
        "format": "minimal_train_snapshot_weights_v1",
        "policy_id": policy_id,
        "update": int(update),
        "device": str(device),
        "config_hash256": config_hash256,
        "structured_policy_contract": structured_policy_contract,
        "model_state_dict": model_state_dict,
        "public_heuristic_logit_bias_scale": public_heuristic_logit_bias_scale,
        "public_heuristic_actor_logit_bias_scale": public_heuristic_actor_logit_bias_scale,
    }
    torch.save(weights_payload, weights_path)
    weights_sha256 = sha256_file(weights_path)

    write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "minimal_train_snapshot_metadata_v1",
            "policy_id": policy_id,
            "update": int(update),
            "weights_path": snapshot_weights_relpath(policy_id),
            "weights_sha256": weights_sha256,
            "source_checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
        },
    )
    return weights_path, weights_sha256


def sync_snapshot_registry_retention(stack: StackWithLeagueConfig, registry: SnapshotRegistry) -> None:
    league = getattr(stack.config, "league", None)
    if league is None:
        return
    registry.recent_size = int(league.snapshot_pool_recent_size)
    registry.champion_size = int(league.snapshot_pool_champion_size)


def snapshot_artifact_dir_for_prune(
    *,
    training_paths: SnapshotTrainingPaths,
    run_dir: Path,
    snapshot: SnapshotMeta,
) -> Path:
    snapshots_root = training_paths.snapshots_dir.resolve()
    weights_path = (run_dir / snapshot.path).resolve()
    try:
        weights_path.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot artifact outside {snapshots_root}: {snapshot.path}") from exc
    if weights_path.name != SNAPSHOT_WEIGHTS_FILENAME:
        raise RuntimeError(f"refusing to delete unexpected snapshot artifact path: {snapshot.path}")

    snapshot_dir = weights_path.parent
    try:
        snapshot_dir.relative_to(snapshots_root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete snapshot directory outside {snapshots_root}: {snapshot_dir}") from exc
    if snapshot_dir == snapshots_root or snapshot_dir.name != snapshot.policy_id:
        raise RuntimeError(f"refusing to delete unexpected snapshot directory: {snapshot_dir}")
    return snapshot_dir


def delete_pruned_snapshot_artifacts(
    *,
    training_paths: SnapshotTrainingPaths,
    run_dir: Path,
    pruned_snapshots: list[SnapshotMeta],
) -> None:
    for snapshot in pruned_snapshots:
        snapshot_dir = snapshot_artifact_dir_for_prune(
            training_paths=training_paths,
            run_dir=run_dir,
            snapshot=snapshot,
        )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)


def save_snapshot_registry_with_retention(
    *,
    stack: StackWithLeagueConfig,
    training_paths: SnapshotTrainingPaths,
    run_dir: Path,
    registry: SnapshotRegistry,
) -> None:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    sync_snapshot_registry_retention(stack, registry)
    pruned_snapshots = registry.prune()
    registry.save(registry_path)
    delete_pruned_snapshot_artifacts(
        training_paths=training_paths,
        run_dir=run_dir,
        pruned_snapshots=pruned_snapshots,
    )


def demote_registry_champions_newer_than(
    training_paths: SnapshotTrainingPaths,
    *,
    update_count: int,
) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    removed = registry.demote_champions_newer_than(int(update_count))
    if removed:
        registry.save(registry_path)
    return removed


def persist_snapshot_registry_entry(
    *,
    stack: StackWithLeagueConfig,
    training_paths: SnapshotTrainingPaths,
    run_dir: Path,
    checkpoint_path: Path,
    model_state_dict: dict[str, Any],
    config_hash256: str,
    device: torch.device,
    update: int,
    policy_version: int,
    public_heuristic_logit_bias_scale: float | None = None,
    public_heuristic_actor_logit_bias_scale: float | None = None,
) -> str:
    policy_id = f"policy_{int(policy_version):06d}"
    model_config = getattr(stack.config, "model", None)
    weights_path, weights_sha256 = write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=policy_id,
        update=update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=model_state_dict,
        structured_policy_contract=None if model_config is None else model_config.structured_policy_contract,
        public_heuristic_logit_bias_scale=public_heuristic_logit_bias_scale,
        public_heuristic_actor_logit_bias_scale=public_heuristic_actor_logit_bias_scale,
    )

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    reg = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, reg)
    reg.add_snapshot(
        policy_id=policy_id,
        update=int(update),
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=reg,
    )
    return policy_id


def seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    source_hash = hashlib.sha1(source_run_dir.as_posix().encode("utf-8")).hexdigest()[:10]
    safe_policy_id = str(source_policy_id).replace("/", "_").replace("\\", "_").strip()
    return f"seed_{source_hash}_{safe_policy_id}"


def write_imported_snapshot_artifact(
    *,
    snapshots_dir: Path,
    run_dir: Path,
    source_payload: dict[str, Any],
    source_run_dir: Path,
    source_policy_id: str,
    source_snapshot_path: str,
    target_policy_id: str,
    update: int,
    metadata_format: str,
    seeded_from_external_registry: bool = False,
    imported_from_update: int | None = None,
) -> tuple[Path, str]:
    snapshot_dir = snapshots_dir / target_policy_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME

    source_weights_path = Path(source_snapshot_path)
    if not source_weights_path.is_absolute():
        source_weights_path = source_run_dir / source_weights_path
    source_weights_sha256 = sha256_file(source_weights_path) if source_weights_path.is_file() else None

    imported_payload = dict(source_payload)
    imported_payload["policy_id"] = target_policy_id
    imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
    imported_payload["imported_from_policy_id"] = source_policy_id
    imported_payload["imported_from_snapshot_path"] = source_snapshot_path
    if source_weights_sha256 is not None:
        imported_payload["imported_from_weights_sha256"] = source_weights_sha256
    if imported_from_update is not None:
        imported_payload["imported_from_update"] = int(imported_from_update)
    if seeded_from_external_registry:
        imported_payload["seeded_from_external_registry"] = True
    torch.save(imported_payload, weights_path)
    weights_sha256 = sha256_file(weights_path)

    metadata: dict[str, Any] = {
        "format": metadata_format,
        "policy_id": target_policy_id,
        "update": int(update),
        "weights_path": snapshot_weights_relpath(target_policy_id),
        "weights_sha256": weights_sha256,
        "imported_from_run_dir": source_run_dir.as_posix(),
        "imported_from_policy_id": source_policy_id,
        "imported_from_snapshot_path": source_snapshot_path,
    }
    if source_weights_sha256 is not None:
        metadata["imported_from_weights_sha256"] = source_weights_sha256
    if imported_from_update is not None:
        metadata["imported_from_update"] = int(imported_from_update)

    write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        metadata,
    )
    return weights_path, weights_sha256
