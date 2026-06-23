from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _import_seed_snapshot_pool_for_test,
    _write_seed_snapshot_run_fixture,
)


def test_import_seed_snapshot_pool_imports_external_snapshots_and_champions(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path)

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run",
    )

    expected_policy_ids = [
        result.train_script._seed_snapshot_policy_id(
            source_run_dir=seed_run_dir.resolve(),
            source_policy_id="policy_000010",
        ),
        result.train_script._seed_snapshot_policy_id(
            source_run_dir=seed_run_dir.resolve(),
            source_policy_id="policy_000020",
        ),
    ]
    assert result.imported_policy_ids == expected_policy_ids

    registry = SnapshotRegistry.load(result.training_paths.snapshots_dir / "registry.json")
    assert [snapshot.policy_id for snapshot in registry.snapshots] == expected_policy_ids
    assert [snapshot.update for snapshot in registry.snapshots] == [0, 0]
    assert registry.champion_snapshots == [expected_policy_ids[-1]]

    weights_path = result.consumer_run_dir / snapshot_weights_relpath(expected_policy_ids[-1])
    metadata_path = result.training_paths.snapshots_dir / expected_policy_ids[-1] / "policy_meta.json"
    assert weights_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "seeded_train_snapshot_metadata_v1"
    assert metadata["policy_id"] == expected_policy_ids[-1]
    assert metadata["update"] == 0
    assert metadata["imported_from_update"] == 20
    assert metadata["imported_from_run_dir"] == seed_run_dir.resolve().as_posix()
    assert metadata["imported_from_policy_id"] == "policy_000020"


def test_import_seed_snapshot_pool_accepts_guided_bootstrap_source_role(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        champion_updates=(),
        experiment_role="guided_league_bootstrap",
    )

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run_guided_bootstrap_seed",
    )

    assert len(result.imported_policy_ids) == 2
