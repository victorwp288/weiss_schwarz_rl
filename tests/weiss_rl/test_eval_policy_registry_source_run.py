from __future__ import annotations

import json
from pathlib import Path

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.eval.simulator.simulator_runner import resolve_eval_policies
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import PolicyValueModel

from ._config_paths import canonical_stack_config_path
from .heuristic_public_test_support import _heuristic_spec_bundle, _write_eval_snapshot, _write_snapshot_registry


def test_resolve_eval_policies_loads_snapshots_from_registry_run_root(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.model is not None

    source_run_dir = tmp_path / "source_run"
    registry_path = source_run_dir / "training" / "snapshots" / "registry.json"
    weights_path = source_run_dir / "training" / "snapshots" / "policy_000100" / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
        observation_spec=_heuristic_spec_bundle()["observation"],  # type: ignore[arg-type]
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "structured_policy_contract": stack.config.model.structured_policy_contract,
        },
        weights_path,
    )

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000100",
        update=100,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000100/weights.pt",
    )
    registry.save(registry_path)

    consumer_run_dir = tmp_path / "consumer_run"
    manifest_path = consumer_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"run_id256": "ab" * 32}), encoding="utf-8")

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=["policy_000100"],
        run_dir=consumer_run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=registry_path,
    )

    assert resolved["policy_000100"].source_run_dir == source_run_dir.resolve().as_posix()
    assert resolved["policy_000100"].snapshot_path == "training/snapshots/policy_000100/weights.pt"
    assert resolved["policy_000100"].model is not None


def test_resolve_eval_policies_accepts_unique_imported_seed_suffix(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    run_dir = tmp_path / "runs" / "seeded_consumer"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    actual_policy_id = "seed_newrun_seed_oldrun_policy_000005"
    requested_policy_id = "seed_oldrun_policy_000005"
    snapshot_weights = _write_eval_snapshot(
        stack=stack,
        run_dir=run_dir,
        policy_id=actual_policy_id,
        update=0,
    )
    _write_snapshot_registry(
        registry_path=registry_path,
        snapshots=[(actual_policy_id, 0, snapshot_weights)],
    )

    resolved = resolve_eval_policies(
        stack=stack,
        policy_ids=[requested_policy_id],
        run_dir=run_dir,
        observation_dim=512,
        action_dim=9,
        spec_bundle=_heuristic_spec_bundle(),
        snapshot_registry_path=registry_path,
    )

    assert resolved[requested_policy_id].policy_id == requested_policy_id
    assert resolved[requested_policy_id].source_run_dir == run_dir.resolve().as_posix()
    assert resolved[requested_policy_id].snapshot_path == f"training/snapshots/{actual_policy_id}/weights.pt"
    assert resolved[requested_policy_id].model is not None
