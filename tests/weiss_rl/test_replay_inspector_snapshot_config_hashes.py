from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.inspector import inspect_replay_bundle

from ._config_paths import canonical_stack_config_path
from .replay_inspector_snapshot_config_test_support import (
    single_step_fake_env,
    write_registry_policy,
    write_replay_run_manifest_and_spec,
    write_single_step_bundle,
)
from .replay_inspector_test_support import _return_fake_env


def test_inspect_replay_bundle_accepts_run_manifest_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    manifest_config_hash = "cd" * 32
    write_replay_run_manifest_and_spec(run_dir=run_dir, config_hash256=manifest_config_hash)
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    registry = SnapshotRegistry()
    write_registry_policy(
        registry=registry,
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        update=1,
        logits={4: 1.0, 9: 0.0},
        config_hash256=manifest_config_hash,
    )
    write_registry_policy(
        registry=registry,
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_b",
        update=2,
        logits={4: 0.0, 9: 1.0},
        config_hash256=manifest_config_hash,
    )
    registry.save(registry_path)

    contract, bundle_path = write_single_step_bundle(tmp_path)
    env = single_step_fake_env(include_terminal_transition=True)

    report = inspect_replay_bundle(
        bundle_path=bundle_path,
        stack=stack,
        run_dir=run_dir,
        snapshot_registry_path=registry_path,
        policy_a="policy_a",
        policy_b="policy_b",
        top_k=1,
        top_actions=2,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
    )

    assert report["compared_steps"] == 1


def test_inspect_replay_bundle_rejects_unmatched_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    write_replay_run_manifest_and_spec(run_dir=run_dir, config_hash256="cd" * 32)
    registry_path = run_dir / "training" / "snapshots" / "registry.json"

    registry = SnapshotRegistry()
    write_registry_policy(
        registry=registry,
        run_dir=run_dir,
        stack=stack,
        policy_id="policy_a",
        update=1,
        logits={4: 1.0, 9: 0.0},
        config_hash256="ef" * 32,
    )
    registry.save(registry_path)

    contract, bundle_path = write_single_step_bundle(tmp_path)
    env = single_step_fake_env(include_terminal_transition=False)

    with pytest.raises(RuntimeError, match="Snapshot config hash mismatch"):
        inspect_replay_bundle(
            bundle_path=bundle_path,
            stack=stack,
            run_dir=run_dir,
            snapshot_registry_path=registry_path,
            policy_a="policy_a",
            policy_b="policy_a",
            top_k=1,
            top_actions=2,
            env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
        )


def test_inspect_replay_bundle_accepts_explicit_extra_snapshot_config_hash(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "run"
    write_replay_run_manifest_and_spec(run_dir=run_dir, config_hash256="cd" * 32)
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    imported_config_hash = "ef" * 32

    registry = SnapshotRegistry()
    write_registry_policy(
        registry=registry,
        run_dir=run_dir,
        stack=stack,
        policy_id="imported_seed",
        update=0,
        logits={4: 1.0, 9: 0.0},
        config_hash256=imported_config_hash,
    )
    registry.save(registry_path)

    contract, bundle_path = write_single_step_bundle(tmp_path)
    env = single_step_fake_env(include_terminal_transition=True)

    report = inspect_replay_bundle(
        bundle_path=bundle_path,
        stack=stack,
        run_dir=run_dir,
        snapshot_registry_path=registry_path,
        policy_a="imported_seed",
        policy_b="imported_seed",
        top_k=1,
        top_actions=2,
        env_factory=lambda observed_contract: _return_fake_env(observed_contract, contract, env),
        accepted_snapshot_config_hashes=[imported_config_hash],
    )

    assert report["compared_steps"] == 1
