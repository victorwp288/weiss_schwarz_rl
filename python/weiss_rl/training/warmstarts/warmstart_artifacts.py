"""Shared artifact writers for auxiliary warmstart workflows."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import canonical_config_dict, compute_config_hash256
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)


def write_warmstart_run_contract_artifacts(
    *,
    output_layout: ArtifactLayout,
    stack: Any,
    source_run_dir: Path | None,
    spec_hash: str,
    manifest_format: str,
    run_kind: str,
) -> None:
    config_hash = compute_config_hash256(stack)
    output_layout.config_hash_path.write_text(config_hash + "\n", encoding="utf-8")
    output_layout.config_json_path.write_text(
        json.dumps(canonical_config_dict(stack), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_layout.spec_hash_path.write_text(str(spec_hash) + "\n", encoding="utf-8")
    manifest = {
        "format": str(manifest_format),
        "run_id256": hashlib.sha256(
            json.dumps(
                {
                    "kind": str(run_kind),
                    "run_dir": output_layout.run_dir.resolve().as_posix(),
                    "config_hash256": config_hash,
                    "spec_hash256": str(spec_hash),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "config_hash256": config_hash,
        "spec_hash256": str(spec_hash),
    }
    output_layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_run_dir is not None:
        source_spec = source_run_dir / "spec_bundle.json"
        if source_spec.is_file():
            shutil.copy2(source_spec, output_layout.spec_bundle_path)


def publish_warmstart_snapshot(
    *,
    output_run_dir: Path,
    checkpoint_path: Path,
    update_count: int,
    policy_id: str,
    metadata_format: str,
) -> dict[str, Any]:
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": str(metadata_format),
        "policy_id": str(policy_id),
        "update": int(update_count),
        "weights_sha256": weights_sha256,
        "source_checkpoint_path": checkpoint_path.resolve().as_posix(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id=str(policy_id),
        update=int(update_count),
        weights_sha256=weights_sha256,
        path=weights_relpath,
    )
    registry.pin_snapshot(str(policy_id))
    registry_path = output_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    registry.save(registry_path)
    return {
        "policy_id": str(policy_id),
        "weights_path": weights_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "registry_path": registry_path.as_posix(),
        "weights_sha256": weights_sha256,
    }


def warmstart_run_contract_writer(*, manifest_format: str, run_kind: str) -> Callable[..., None]:
    def write_run_contract_artifacts(
        *,
        output_layout: ArtifactLayout,
        stack: Any,
        source_run_dir: Path | None,
        spec_hash: str,
    ) -> None:
        write_warmstart_run_contract_artifacts(
            output_layout=output_layout,
            stack=stack,
            source_run_dir=source_run_dir,
            spec_hash=spec_hash,
            manifest_format=manifest_format,
            run_kind=run_kind,
        )

    write_run_contract_artifacts.__name__ = f"write_{run_kind}_contract_artifacts"
    return write_run_contract_artifacts


def warmstart_snapshot_publisher(*, policy_id: str, metadata_format: str) -> Callable[..., dict[str, Any]]:
    def publish_snapshot(*, output_run_dir: Path, checkpoint_path: Path, update_count: int) -> dict[str, Any]:
        return publish_warmstart_snapshot(
            output_run_dir=output_run_dir,
            checkpoint_path=checkpoint_path,
            update_count=update_count,
            policy_id=policy_id,
            metadata_format=metadata_format,
        )

    publish_snapshot.__name__ = f"publish_{policy_id}_snapshot"
    return publish_snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "publish_warmstart_snapshot",
    "sha256_file",
    "warmstart_run_contract_writer",
    "warmstart_snapshot_publisher",
    "write_warmstart_run_contract_artifacts",
]
