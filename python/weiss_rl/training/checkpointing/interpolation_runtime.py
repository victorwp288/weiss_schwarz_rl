"""Runtime orchestration for checkpoint interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.training.checkpointing.interpolation import interpolate_model_state_dicts


@dataclass(frozen=True, slots=True)
class CheckpointInterpolationRunResult:
    checkpoint_path: Path
    summary_path: Path
    summary: dict[str, Any]


def run_checkpoint_interpolation(args: argparse.Namespace) -> CheckpointInterpolationRunResult:
    first_payload = load_checkpoint(args.first_checkpoint)
    second_payload = load_checkpoint(args.second_checkpoint)
    validate_checkpoint_contracts(
        first_payload,
        second_payload,
        allow_config_hash_mismatch=bool(args.allow_config_hash_mismatch),
    )
    mixed_state = interpolate_model_state_dicts(
        model_state_dict(first_payload, args.first_checkpoint),
        model_state_dict(second_payload, args.second_checkpoint),
        second_weight=float(args.second_weight),
    )

    output_layout = ArtifactLayout.from_run_dir(args.output_run_dir)
    output_layout.ensure_directories()
    copy_contract_artifacts(
        source_run_dir=args.second_run_dir,
        output_layout=output_layout,
        first_run_dir=args.first_run_dir,
        first_checkpoint=args.first_checkpoint,
        second_checkpoint=args.second_checkpoint,
        second_weight=float(args.second_weight),
    )
    checkpoint_path = output_layout.run_dir / "training" / "checkpoints" / "checkpoint_interpolated.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(second_payload)
    payload["model_state_dict"] = mixed_state
    payload["policy_anchor_model_state_dict"] = None
    payload["optimizer_state_dict"] = None
    payload["grad_scaler_state_dict"] = None
    payload["interpolation"] = {
        "format": "checkpoint_interpolation_v1",
        "first_checkpoint": args.first_checkpoint.resolve().as_posix(),
        "second_checkpoint": args.second_checkpoint.resolve().as_posix(),
        "second_weight": float(args.second_weight),
    }
    torch.save(payload, checkpoint_path)
    shutil.copy2(checkpoint_path, output_layout.run_dir / "training" / "checkpoints" / "latest.pt")
    snapshot_payload = publish_interpolated_snapshot(
        output_run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=str(args.policy_id),
        update_count=int(payload.get("update_count", 0)),
    )
    summary = {
        "format": "checkpoint_interpolation_summary_v1",
        "first_checkpoint": args.first_checkpoint.resolve().as_posix(),
        "second_checkpoint": args.second_checkpoint.resolve().as_posix(),
        "output_run_dir": output_layout.run_dir.resolve().as_posix(),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "second_weight": float(args.second_weight),
        "snapshot": snapshot_payload,
    }
    summary_path = output_layout.run_dir / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CheckpointInterpolationRunResult(
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        summary=summary,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {path}")
    return payload


def model_state_dict(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"checkpoint is missing model_state_dict: {path}")
    return state


def validate_checkpoint_contracts(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    allow_config_hash_mismatch: bool = False,
) -> None:
    for field in ("format", "config_hash256", "spec_hash256", "algorithm", "recurrent_core"):
        if field == "config_hash256" and allow_config_hash_mismatch:
            continue
        if first.get(field) != second.get(field):
            raise RuntimeError(f"checkpoint field {field!r} differs: {first.get(field)!r} != {second.get(field)!r}")


def copy_contract_artifacts(
    *,
    source_run_dir: Path,
    output_layout: ArtifactLayout,
    first_run_dir: Path,
    first_checkpoint: Path,
    second_checkpoint: Path,
    second_weight: float,
) -> None:
    source = Path(source_run_dir)
    for name in (
        "config_canonical.json",
        "config_hash256.txt",
        "spec_hash256.txt",
        "spec_bundle.json",
        "config.json",
        "config_hash.txt",
        "spec_hash.txt",
    ):
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, output_layout.run_dir / name)
    source_manifest = source / "manifest.json"
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update(
        {
            "format": "checkpoint_interpolation_manifest_v1",
            "run_id256": hashlib.sha256(
                json.dumps(
                    {
                        "kind": "checkpoint_interpolation",
                        "first_run_dir": Path(first_run_dir).resolve().as_posix(),
                        "source_run_dir": source.resolve().as_posix(),
                        "first_checkpoint": Path(first_checkpoint).resolve().as_posix(),
                        "second_checkpoint": Path(second_checkpoint).resolve().as_posix(),
                        "second_weight": float(second_weight),
                        "created_unix": time.time(),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "interpolation": {
                "first_run_dir": Path(first_run_dir).resolve().as_posix(),
                "second_run_dir": source.resolve().as_posix(),
                "first_checkpoint": Path(first_checkpoint).resolve().as_posix(),
                "second_checkpoint": Path(second_checkpoint).resolve().as_posix(),
                "second_weight": float(second_weight),
            },
        }
    )
    output_layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_interpolated_snapshot(
    *,
    output_run_dir: Path,
    checkpoint_path: Path,
    policy_id: str,
    update_count: int,
) -> dict[str, Any]:
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "interpolated_snapshot_meta_v1",
        "policy_id": policy_id,
        "update": int(update_count),
        "weights_sha256": weights_sha256,
        "source_checkpoint_path": checkpoint_path.resolve().as_posix(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id=policy_id,
        update=int(update_count),
        weights_sha256=weights_sha256,
        path=weights_relpath,
    )
    registry.pin_snapshot(policy_id)
    registry_path = output_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    registry.save(registry_path)
    return {
        "policy_id": policy_id,
        "weights_path": weights_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "registry_path": registry_path.as_posix(),
        "weights_sha256": weights_sha256,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CheckpointInterpolationRunResult",
    "copy_contract_artifacts",
    "load_checkpoint",
    "model_state_dict",
    "publish_interpolated_snapshot",
    "run_checkpoint_interpolation",
    "sha256_file",
    "validate_checkpoint_contracts",
]
