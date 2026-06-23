from __future__ import annotations

from pathlib import Path

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import _load_train_script_module, _make_bootstrap_learner


def test_ensure_noleague_baseline_anchor_refreshes_current_run_alias_to_latest_update(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(canonical_stack_config_path())
    run_dir = tmp_path / "b1_run"
    training_paths = train_script._training_paths(run_dir)
    learner = _make_bootstrap_learner(stack, update_count=1, policy_version=1)

    first_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        permit_current_run_alias=True,
        update=1,
    )

    first_registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    first_snapshot = next(snapshot for snapshot in first_registry.snapshots if snapshot.policy_id == first_policy_id)
    first_hash = first_snapshot.weights_sha256

    learner.update_count = 3
    second_policy_id = train_script._ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        permit_current_run_alias=True,
        update=3,
    )

    assert second_policy_id == first_policy_id == "b1_noleague_baseline"
    second_registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    second_snapshot = next(snapshot for snapshot in second_registry.snapshots if snapshot.policy_id == second_policy_id)
    assert second_snapshot.update == 3
    assert second_snapshot.weights_sha256 != first_hash
