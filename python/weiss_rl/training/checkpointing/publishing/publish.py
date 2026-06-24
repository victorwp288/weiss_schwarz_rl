from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from weiss_rl.league.registry import REGISTRY_FILENAME, SNAPSHOT_METADATA_FILENAME, SnapshotRegistry
from weiss_rl.training.snapshots import write_snapshot_artifact

CHECKPOINT_SNAPSHOT_METADATA_FORMAT = "checkpoint_candidate_snapshot_metadata_v1"


def default_checkpoint_snapshot_policy_id(update: int) -> str:
    return f"checkpoint_{int(update):06d}"


def publish_checkpoint_snapshot(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    policy_id: str | None = None,
    pin: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    checkpoint_path = _resolve_checkpoint_path(run_dir=run_dir, checkpoint_path=checkpoint_path)
    payload = _load_checkpoint_payload(checkpoint_path)
    update = _checkpoint_update(payload, checkpoint_path=checkpoint_path)
    resolved_policy_id = str(policy_id or default_checkpoint_snapshot_policy_id(update)).strip()
    if not resolved_policy_id:
        raise ValueError("policy_id must be non-empty")

    snapshots_dir = run_dir / "training" / "snapshots"
    registry_path = snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    existing = next((snapshot for snapshot in registry.snapshots if snapshot.policy_id == resolved_policy_id), None)
    if existing is not None and not replace:
        if int(existing.update) != int(update):
            raise ValueError(
                f"snapshot {resolved_policy_id!r} already exists at update {existing.update}, "
                f"not checkpoint update {update}"
            )
        metadata_path = run_dir / "training" / "snapshots" / resolved_policy_id / SNAPSHOT_METADATA_FILENAME
        return {
            "policy_id": resolved_policy_id,
            "update": int(existing.update),
            "weights_sha256": existing.weights_sha256,
            "weights_path": existing.path,
            "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
            "registry_path": registry_path.as_posix(),
            "source_checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
            "already_exists": True,
            "pinned": resolved_policy_id in registry.pinned_snapshots,
        }

    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")
    config_hash256 = str(payload.get("config_hash256", "")).strip()
    if not config_hash256:
        raise ValueError(f"checkpoint is missing config_hash256: {checkpoint_path}")

    weights_path, weights_sha256 = write_snapshot_artifact(
        snapshots_dir=snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=resolved_policy_id,
        update=update,
        config_hash256=config_hash256,
        device=torch.device(str(payload.get("device") or "cpu")),
        model_state_dict=model_state_dict,
        structured_policy_contract=_manifest_structured_policy_contract(run_dir),
        public_heuristic_logit_bias_scale=_optional_float(payload.get("public_heuristic_logit_bias_scale")),
        public_heuristic_actor_logit_bias_scale=_optional_float(payload.get("public_heuristic_actor_logit_bias_scale")),
    )
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = _json_object(metadata_path)
    metadata.update(
        {
            "format": CHECKPOINT_SNAPSHOT_METADATA_FORMAT,
            "source_checkpoint_update": update,
            "source_checkpoint_policy_version": payload.get("policy_version"),
            "source_checkpoint_format": payload.get("format"),
            "source_checkpoint_spec_hash256": payload.get("spec_hash256"),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry.add_snapshot(
        policy_id=resolved_policy_id,
        update=update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    if pin:
        registry.pin_snapshot(resolved_policy_id)
    registry.save(registry_path)
    return {
        "policy_id": resolved_policy_id,
        "update": int(update),
        "weights_sha256": weights_sha256,
        "weights_path": weights_path.relative_to(run_dir).as_posix(),
        "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
        "registry_path": registry_path.as_posix(),
        "source_checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
        "already_exists": False,
        "pinned": bool(pin),
    }


def _resolve_checkpoint_path(*, run_dir: Path, checkpoint_path: Path) -> Path:
    candidate = Path(checkpoint_path)
    if not candidate.is_absolute():
        cwd_relative = candidate.resolve()
        candidate = cwd_relative if cwd_relative.is_file() else run_dir / candidate
    resolved = candidate.resolve()
    checkpoint_root = (run_dir / "training" / "checkpoints").resolve()
    try:
        relative = resolved.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(f"checkpoint must be under {checkpoint_root}: {resolved}") from exc
    if len(relative.parts) != 1 or not relative.name.startswith("checkpoint_") or relative.suffix != ".pt":
        raise ValueError("checkpoint publication requires a numbered training/checkpoints/checkpoint_<update>.pt file")
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict: {checkpoint_path}")
    checkpoint_format = str(payload.get("format", "")).strip()
    if checkpoint_format != "minimal_train_checkpoint_v1":
        raise ValueError(f"unsupported checkpoint format {checkpoint_format!r}: {checkpoint_path}")
    return payload


def _checkpoint_update(payload: dict[str, Any], *, checkpoint_path: Path) -> int:
    raw_update = payload.get("update_count")
    if isinstance(raw_update, bool) or not isinstance(raw_update, int):
        raise ValueError(f"checkpoint is missing integer update_count: {checkpoint_path}")
    update = int(raw_update)
    if update < 0:
        raise ValueError(f"checkpoint update_count must be >= 0: {checkpoint_path}")
    expected_name = f"checkpoint_{update}.pt"
    if checkpoint_path.name != expected_name:
        raise ValueError(f"checkpoint filename {checkpoint_path.name!r} does not match update_count {update}")
    return update


def _manifest_structured_policy_contract(run_dir: Path) -> str | None:
    manifest = _json_object(run_dir / "manifest.json")
    config = manifest.get("config_canonical", {}).get("config", {})
    if not isinstance(config, dict):
        return None
    model = config.get("model", {})
    if not isinstance(model, dict):
        return None
    contract = model.get("structured_policy_contract")
    return str(contract) if isinstance(contract, str) and contract.strip() else None


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
