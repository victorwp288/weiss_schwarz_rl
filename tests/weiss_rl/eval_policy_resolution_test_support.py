from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.config import canonical_config_dict

from .heuristic_public_test_support import _write_eval_snapshot, _write_snapshot_registry


def write_consumer_run_manifest(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / "runs" / name
    (run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")
    return run_dir


def write_snapshot_registry_run(
    *,
    stack: object,
    run_dir: Path,
    snapshots: list[tuple[str, int]],
) -> dict[str, Path]:
    weights_by_policy_id: dict[str, Path] = {}
    registry_snapshots: list[tuple[str, int, Path]] = []
    for policy_id, update in snapshots:
        weights_path = _write_eval_snapshot(
            stack=stack,
            run_dir=run_dir,
            policy_id=policy_id,
            update=update,
        )
        weights_by_policy_id[policy_id] = weights_path
        registry_snapshots.append((policy_id, update, weights_path))
    _write_snapshot_registry(
        registry_path=run_dir / "training" / "snapshots" / "registry.json",
        snapshots=registry_snapshots,
    )
    return weights_by_policy_id


def copy_snapshot_registry_to_cache(registry_path: Path, tmp_path: Path) -> Path:
    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    return copied_registry_path


def copy_policy_weights_to_cache(*, source_weights: Path, tmp_path: Path, policy_id: str) -> Path:
    copied_policy_weights = tmp_path / "cache" / "training" / "snapshots" / policy_id / "weights.pt"
    copied_policy_weights.parent.mkdir(parents=True, exist_ok=True)
    copied_policy_weights.write_bytes(source_weights.read_bytes())
    return copied_policy_weights


def write_nested_b1_manifest_marker(*, stack: object, run_dir: Path) -> None:
    config_canonical = canonical_config_dict(stack)
    config_canonical["config"]["experiment"] = {
        **dict(config_canonical["config"].get("experiment", {})),
        "role": "baseline_noleague",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_canonical}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_legacy_b1_manifest_marker(*, stack: object, run_dir: Path) -> None:
    config_sections = dict(cast(dict[str, Any], canonical_config_dict(stack).get("config", {})))
    config_sections.pop("experiment", None)
    config_sections["training_family_a"] = {"mode": "b1_no_league"}
    (run_dir / "manifest.json").write_text(
        json.dumps({"config_canonical": config_sections}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
