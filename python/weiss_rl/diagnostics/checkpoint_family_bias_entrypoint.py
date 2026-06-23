"""Create a diagnostic checkpoint with explicit action-family logit offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.runtime.components.legal_meta import action_catalog_indices

_FAMILY_BIAS_KEY = "policy_head.family_bias"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy a checkpoint and add explicit action-family logit biases")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument(
        "--family-bias",
        action="append",
        default=[],
        metavar="FAMILY=DELTA",
        help="Action-family logit delta to add. Repeat for multiple families.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_payload = _load_checkpoint(args.source_checkpoint)
    source_run_dir = Path(args.source_run_dir)
    output_layout = ArtifactLayout.from_run_dir(args.output_run_dir)
    output_layout.ensure_directories()

    spec_bundle = _load_spec_bundle(source_run_dir)
    family_index, _attack_type_index = action_catalog_indices(ActionCatalog.from_spec_bundle(spec_bundle))
    family_offsets = _parse_family_bias_offsets(args.family_bias)
    changed_indices = _family_bias_indices(family_offsets, family_index=family_index)

    payload = dict(source_payload)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, MutableMapping):
        raise RuntimeError(f"checkpoint is missing model_state_dict: {args.source_checkpoint}")
    payload["model_state_dict"] = _apply_family_bias_offsets(
        model_state,
        family_offsets=family_offsets,
        family_index=family_index,
    )
    payload["optimizer_state_dict"] = None
    payload["grad_scaler_state_dict"] = None
    payload["family_logit_bias_surgery"] = {
        "format": "family_logit_bias_surgery_v1",
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "family_offsets": dict(sorted(family_offsets.items())),
        "family_indices": changed_indices,
    }

    _copy_contract_artifacts(source_run_dir=source_run_dir, output_layout=output_layout, args=args)
    checkpoint_path = output_layout.run_dir / "training" / "checkpoints" / "checkpoint_family_bias.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    shutil.copy2(checkpoint_path, output_layout.run_dir / "training" / "checkpoints" / "latest.pt")
    snapshot_payload = _publish_snapshot(
        output_run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=str(args.policy_id),
        update_count=int(payload.get("update_count", 0)),
        family_offsets=family_offsets,
    )
    summary = {
        "format": "family_logit_bias_surgery_summary_v1",
        "source_checkpoint": args.source_checkpoint.resolve().as_posix(),
        "source_run_dir": source_run_dir.resolve().as_posix(),
        "output_run_dir": output_layout.run_dir.resolve().as_posix(),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "family_offsets": dict(sorted(family_offsets.items())),
        "family_indices": changed_indices,
        "snapshot": snapshot_payload,
    }
    summary_path = output_layout.run_dir / "eval" / "diagnostics" / "family_logit_bias_surgery_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Family-bias checkpoint written to {checkpoint_path}; policy_id={args.policy_id}; summary={summary_path}")
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {path}")
    return payload


def _load_spec_bundle(run_dir: Path) -> Mapping[str, object]:
    path = Path(run_dir) / "spec_bundle.json"
    if not path.is_file():
        raise FileNotFoundError(f"spec_bundle.json not found in source run: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"spec_bundle.json must contain an object: {path}")
    return payload


def _parse_family_bias_offsets(items: Sequence[str]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for item in items:
        raw = str(item).strip()
        if not raw or "=" not in raw:
            raise ValueError(f"family bias must be FAMILY=DELTA, got {item!r}")
        family, raw_delta = raw.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError(f"family bias must name a family, got {item!r}")
        try:
            delta = float(raw_delta)
        except ValueError as exc:
            raise ValueError(f"family bias delta must be a float, got {item!r}") from exc
        offsets[family] = offsets.get(family, 0.0) + float(delta)
    if not offsets:
        raise ValueError("at least one --family-bias FAMILY=DELTA item is required")
    return offsets


def _family_bias_indices(
    family_offsets: Mapping[str, float],
    *,
    family_index: Mapping[str, int],
) -> dict[str, int]:
    missing = [family for family in family_offsets if family not in family_index]
    if missing:
        raise ValueError("unknown action families in --family-bias: " + ", ".join(sorted(missing)))
    return {family: int(family_index[family]) for family in sorted(family_offsets)}


def _apply_family_bias_offsets(
    model_state: Mapping[str, Any],
    *,
    family_offsets: Mapping[str, float],
    family_index: Mapping[str, int],
) -> dict[str, Any]:
    if _FAMILY_BIAS_KEY not in model_state:
        raise RuntimeError(f"model_state_dict is missing {_FAMILY_BIAS_KEY!r}")
    result = dict(model_state)
    family_bias = model_state[_FAMILY_BIAS_KEY]
    if not isinstance(family_bias, torch.Tensor) or not family_bias.is_floating_point():
        raise RuntimeError(f"{_FAMILY_BIAS_KEY} must be a floating tensor")
    changed = family_bias.detach().clone()
    for family, delta in family_offsets.items():
        if family not in family_index:
            raise ValueError(f"unknown action family in --family-bias: {family}")
        index = int(family_index[family])
        if index < 0 or index >= int(changed.numel()):
            raise RuntimeError(f"family index for {family!r} is outside {_FAMILY_BIAS_KEY}: {index}")
        changed[index] = changed[index] + float(delta)
    result[_FAMILY_BIAS_KEY] = changed
    return result


def _copy_contract_artifacts(
    *,
    source_run_dir: Path,
    output_layout: ArtifactLayout,
    args: argparse.Namespace,
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
            "format": "family_logit_bias_surgery_manifest_v1",
            "run_id256": hashlib.sha256(
                json.dumps(
                    {
                        "kind": "family_logit_bias_surgery",
                        "source_run_dir": source.resolve().as_posix(),
                        "source_checkpoint": Path(args.source_checkpoint).resolve().as_posix(),
                        "policy_id": str(args.policy_id),
                        "family_bias": list(args.family_bias),
                        "created_unix": time.time(),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "family_logit_bias_surgery": {
                "source_run_dir": source.resolve().as_posix(),
                "source_checkpoint": Path(args.source_checkpoint).resolve().as_posix(),
                "policy_id": str(args.policy_id),
                "family_bias": list(args.family_bias),
            },
        }
    )
    output_layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publish_snapshot(
    *,
    output_run_dir: Path,
    checkpoint_path: Path,
    policy_id: str,
    update_count: int,
    family_offsets: Mapping[str, float],
) -> dict[str, Any]:
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = _sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "family_logit_bias_surgery_snapshot_meta_v1",
        "policy_id": policy_id,
        "update": int(update_count),
        "weights_sha256": weights_sha256,
        "source_checkpoint_path": checkpoint_path.resolve().as_posix(),
        "family_offsets": dict(sorted(family_offsets.items())),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
