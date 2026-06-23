from __future__ import annotations

import json
from pathlib import Path

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    REPO_ROOT,
    _load_train_script_module,
    _make_bootstrap_learner,
    _mark_fixture_as_locked_selected_candidate,
    _write_b1_baseline_run_fixture,
)


def test_ensure_noleague_baseline_anchor_imports_frozen_snapshot_once(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, update=5)

    run_dir = tmp_path / "consumer_run"
    training_paths = train_script._training_paths(run_dir)
    bootstrap_learner = _make_bootstrap_learner(stack)

    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=bootstrap_learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry_path = training_paths.snapshots_dir / "registry.json"
    registry = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [policy_id]
    assert registry.champion_snapshots == []
    assert registry.pinned_snapshots == [policy_id]

    snapshot = registry.snapshots[0]
    weights_path = run_dir / snapshot_weights_relpath(policy_id)
    metadata_path = training_paths.snapshots_dir / policy_id / "policy_meta.json"

    assert snapshot.update == 5
    assert weights_path.is_file()
    assert snapshot.weights_sha256 == train_script._sha256_file(weights_path)
    source_weights_sha256 = train_script._sha256_file(baseline_run_dir / snapshot_weights_relpath(policy_id))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "imported_train_snapshot_metadata_v1",
        "imported_from_policy_id": policy_id,
        "imported_from_run_dir": baseline_run_dir.resolve().as_posix(),
        "imported_from_snapshot_path": snapshot_weights_relpath(policy_id),
        "imported_from_weights_sha256": source_weights_sha256,
        "policy_id": policy_id,
        "update": 5,
        "weights_path": snapshot_weights_relpath(policy_id),
        "weights_sha256": snapshot.weights_sha256,
    }

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert payload["policy_id"] == policy_id
    assert payload["update"] == 5
    assert payload["imported_from_run_dir"] == baseline_run_dir.resolve().as_posix()
    assert payload["imported_from_policy_id"] == policy_id
    assert payload["imported_from_snapshot_path"] == snapshot_weights_relpath(policy_id)
    assert payload["imported_from_weights_sha256"] == source_weights_sha256

    second_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=bootstrap_learner,
        device=torch.device("cpu"),
        config_hash256="ff" * 32,
        baseline_run_dir=baseline_run_dir,
    )

    assert second_policy_id == policy_id
    reloaded = SnapshotRegistry.load(registry_path)
    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [policy_id]
    assert reloaded.champion_snapshots == []
    assert reloaded.pinned_snapshots == [policy_id]
    assert reloaded.snapshots[0].weights_sha256 == snapshot.weights_sha256


def test_ensure_noleague_baseline_anchor_imports_locked_selected_candidate(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
    )
    _mark_fixture_as_locked_selected_candidate(selected_run_dir, update=15)

    run_dir = tmp_path / "consumer_selected_run"
    training_paths = train_script._training_paths(run_dir)
    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=_make_bootstrap_learner(stack),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=selected_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    snapshot = next(entry for entry in registry.snapshots if entry.policy_id == policy_id)
    assert snapshot.update == 15
    payload = torch.load(run_dir / snapshot_weights_relpath(policy_id), map_location="cpu", weights_only=True)
    assert payload["imported_from_policy_id"] == "selected_candidate"
    assert payload["imported_from_run_dir"] == selected_run_dir.resolve().as_posix()
    assert payload["imported_from_weights_sha256"] == train_script._sha256_file(
        selected_run_dir / snapshot_weights_relpath("selected_candidate")
    )


def test_ensure_noleague_baseline_anchor_imports_explicit_b1_run_when_required_by_main_gate(
    tmp_path: Path,
) -> None:
    train_script = _load_train_script_module()
    stack_path = REPO_ROOT / "configs" / "thesis" / "main_league.yaml"
    stack = load_stack_config(stack_path)
    league_config = stack.config.league
    assert league_config is not None
    assert "B1 NoLeague baseline" in league_config.promotion_anchor_set_v1.required
    selected_run_dir = _write_b1_baseline_run_fixture(
        tmp_path,
        update=15,
        policy_id="selected_candidate",
        experiment_role="guided_league_bootstrap",
        stack_path=stack_path,
    )
    _mark_fixture_as_locked_selected_candidate(selected_run_dir, update=15)

    run_dir = tmp_path / "consumer_optional_b1_run"
    training_paths = train_script._training_paths(run_dir)
    policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=_make_bootstrap_learner(stack),
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        baseline_run_dir=selected_run_dir,
    )

    assert policy_id == "b1_noleague_baseline"
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert registry.pinned_snapshots == [policy_id]
    payload = torch.load(run_dir / snapshot_weights_relpath(policy_id), map_location="cpu", weights_only=True)
    assert payload["imported_from_policy_id"] == "selected_candidate"
    assert payload["imported_from_weights_sha256"] == train_script._sha256_file(
        selected_run_dir / snapshot_weights_relpath("selected_candidate")
    )
