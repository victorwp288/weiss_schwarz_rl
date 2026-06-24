from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.eval.simulator.simulator_runner import resolve_eval_policies

from ._config_paths import canonical_stack_config_path
from .heuristic_public_test_support import _heuristic_spec_bundle, _write_eval_snapshot, _write_snapshot_registry


def test_resolve_eval_policies_loads_snapshots_from_copied_registry_json(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_run_dir = tmp_path / "runs" / "consumer_run"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_prefers_explicit_run_dir_for_ambiguous_copied_registry(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_run_dir = tmp_path / "runs" / "copied_run"
    copied_weights_path = copied_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    copied_weights_path.parent.mkdir(parents=True, exist_ok=True)
    copied_weights_path.write_bytes(snapshot_weights.read_bytes())

    copied_registry_path = tmp_path / "cache" / "policy_set_snapshot_registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=copied_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == copied_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_ignores_canonical_looking_copied_registry_without_weights(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    source_run_dir = tmp_path / "external_runs" / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=source_run_dir,
        policy_id="policy_000100",
        update=100,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[("policy_000100", 100, snapshot_weights)],
    )

    copied_registry_path = tmp_path / "cache" / "training" / "snapshots" / "registry.json"
    copied_registry_path.parent.mkdir(parents=True, exist_ok=True)
    copied_registry_path.write_text(registry_path.read_text(encoding="utf-8"), encoding="utf-8")
    consumer_run_dir = tmp_path / "runs" / "consumer_run_canonical_copy"
    (consumer_run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (consumer_run_dir / "manifest.json").write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=copied_registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None
