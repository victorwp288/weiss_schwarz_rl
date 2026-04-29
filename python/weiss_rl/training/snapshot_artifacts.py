"""Snapshot artifact and registry persistence helpers for training runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from weiss_rl.config import StackConfig
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
    snapshot_weights_relpath,
)


class SnapshotArtifactPaths(Protocol):
    snapshots_dir: Path


@dataclass(frozen=True, slots=True)
class PromotionGateRegistryUpdate:
    passed: bool
    candidate_policy_id: str
    update_count: int
    ordered_opponents: tuple[str, ...]
    reason_codes: str
    registry_updated: bool


def format_promotion_gate_registry_update_message(update: PromotionGateRegistryUpdate) -> str:
    if update.passed:
        return (
            "Promotion gate passed: "
            f"update={update.update_count} candidate={update.candidate_policy_id} "
            f"anchors={','.join(update.ordered_opponents)}"
        )
    return (
        "Promotion gate failed: "
        f"update={update.update_count} candidate={update.candidate_policy_id} "
        f"reasons={update.reason_codes}"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    public_heuristic_logit_bias_scale: float | None = None,
    public_heuristic_actor_logit_bias_scale: float | None = None,
) -> tuple[Path, str]:
    snapshot_dir = snapshots_dir / policy_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    weights_path = snapshot_dir / "weights.pt"
    weights_payload = {
        "format": "minimal_train_snapshot_weights_v1",
        "policy_id": policy_id,
        "update": int(update),
        "device": str(device),
        "config_hash256": config_hash256,
        "model_state_dict": model_state_dict,
        "public_heuristic_logit_bias_scale": public_heuristic_logit_bias_scale,
        "public_heuristic_actor_logit_bias_scale": public_heuristic_actor_logit_bias_scale,
    }
    torch.save(weights_payload, weights_path)
    weights_sha256 = sha256_file(weights_path)

    _write_json_file(
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


def sync_snapshot_registry_retention(stack: StackConfig, registry: SnapshotRegistry) -> None:
    league = stack.config.league
    if league is None:
        return
    registry.recent_size = int(league.snapshot_pool_recent_size)
    registry.champion_size = int(league.snapshot_pool_champion_size)


def snapshot_artifact_dir_for_prune(
    *,
    training_paths: SnapshotArtifactPaths,
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
    training_paths: SnapshotArtifactPaths,
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
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
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


def pin_snapshot_ids(
    *,
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
    run_dir: Path,
    snapshot_ids: Sequence[str],
) -> tuple[str, ...]:
    requested_ids = tuple(
        dict.fromkeys(str(snapshot_id).strip() for snapshot_id in snapshot_ids if str(snapshot_id).strip())
    )
    if not requested_ids:
        return ()
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    existing_pins = set(registry.pinned_snapshots)
    newly_pinned: list[str] = []
    for snapshot_id in requested_ids:
        registry.pin_snapshot(snapshot_id)
        if snapshot_id not in existing_pins:
            newly_pinned.append(snapshot_id)
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    return tuple(newly_pinned)


def unpin_snapshot_ids(
    *,
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
    run_dir: Path,
    snapshot_ids: Sequence[str],
) -> None:
    removable_ids = {str(snapshot_id).strip() for snapshot_id in snapshot_ids if str(snapshot_id).strip()}
    if not removable_ids:
        return
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    registry.pinned_snapshots = [
        snapshot_id for snapshot_id in registry.pinned_snapshots if snapshot_id not in removable_ids
    ]
    registry.normalize()
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )


def apply_promotion_gate_payload(
    *,
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
    run_dir: Path,
    payload: Mapping[str, Any],
) -> PromotionGateRegistryUpdate:
    candidate_policy_id = str(payload["candidate_policy_id"])
    update_count = int(payload["update_count"])
    ordered_opponents = tuple(str(opponent) for opponent in payload.get("ordered_opponents", ()))
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)

    if bool(payload["passed"]):
        registry.add_champion(candidate_policy_id)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return PromotionGateRegistryUpdate(
            passed=True,
            candidate_policy_id=candidate_policy_id,
            update_count=update_count,
            ordered_opponents=ordered_opponents,
            reason_codes="",
            registry_updated=True,
        )

    raw_reasons = payload.get("reasons", ())
    reason_codes = (
        ",".join(
            str(reason.get("code", "unknown")) if isinstance(reason, Mapping) else "unknown" for reason in raw_reasons
        )
        or "unknown"
    )
    registry_updated = False
    if registry.has_snapshot(candidate_policy_id):
        registry.reject_snapshot(candidate_policy_id)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        registry_updated = True
    return PromotionGateRegistryUpdate(
        passed=False,
        candidate_policy_id=candidate_policy_id,
        update_count=update_count,
        ordered_opponents=ordered_opponents,
        reason_codes=reason_codes,
        registry_updated=registry_updated,
    )


def apply_promotion_gate_result(
    *,
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
    run_dir: Path,
    registry: SnapshotRegistry,
    candidate_policy_id: str,
    update_count: int,
    result: Any,
) -> PromotionGateRegistryUpdate:
    ordered_opponents = tuple(str(opponent) for opponent in result.ordered_opponents)
    if bool(result.passed):
        registry.add_champion(candidate_policy_id)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return PromotionGateRegistryUpdate(
            passed=True,
            candidate_policy_id=str(candidate_policy_id),
            update_count=int(update_count),
            ordered_opponents=ordered_opponents,
            reason_codes="",
            registry_updated=True,
        )

    reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
    registry.reject_snapshot(candidate_policy_id)
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    return PromotionGateRegistryUpdate(
        passed=False,
        candidate_policy_id=str(candidate_policy_id),
        update_count=int(update_count),
        ordered_opponents=ordered_opponents,
        reason_codes=reason_codes,
        registry_updated=True,
    )


def persist_snapshot_registry_entry(
    *,
    stack: StackConfig,
    training_paths: SnapshotArtifactPaths,
    run_dir: Path,
    checkpoint_path: Path,
    model_state_dict: dict[str, Any],
    config_hash256: str,
    device: torch.device,
    update: int,
    policy_version: int,
    guidance_payload: Mapping[str, Any] | None = None,
) -> str:
    policy_id = f"policy_{int(policy_version):06d}"
    guidance = guidance_payload or {}
    weights_path, weights_sha256 = write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=policy_id,
        update=update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=model_state_dict,
        public_heuristic_logit_bias_scale=guidance.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance.get("public_heuristic_actor_logit_bias_scale"),
    )

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    registry.add_snapshot(
        policy_id=policy_id,
        update=int(update),
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    return policy_id
