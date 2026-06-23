from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry

from ._config_paths import canonical_stack_config_path
from .snapshot_registry_test_support import (
    _import_seed_snapshot_pool_for_test,
    _write_seed_snapshot_run_fixture,
)


def test_import_seed_snapshot_pool_can_mark_all_seed_snapshots_as_training_champions(tmp_path: Path) -> None:
    stack = _stack_with_seed_pool(tmp_path, seed_snapshot_champion_import="all")
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, champion_updates=())

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run_seedchampions",
    )

    registry = SnapshotRegistry.load(result.training_paths.snapshots_dir / "registry.json")
    assert result.imported_policy_ids
    assert registry.champion_snapshots == result.imported_policy_ids


def test_import_seed_snapshot_pool_can_mark_pinned_seed_snapshots_as_training_champions(tmp_path: Path) -> None:
    stack = _stack_with_seed_pool(tmp_path, seed_snapshot_champion_import="pinned")
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(),
        pinned_policy_ids=("policy_000020",),
    )

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run_pinned_seedchampions",
    )

    expected_champion = result.train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000020",
    )
    registry = SnapshotRegistry.load(result.training_paths.snapshots_dir / "registry.json")
    assert len(result.imported_policy_ids) == 3
    assert registry.champion_snapshots == [expected_champion]


def test_import_seed_snapshot_pool_can_import_only_pinned_seed_snapshots(tmp_path: Path) -> None:
    stack = _stack_with_seed_pool(
        tmp_path,
        seed_snapshot_champion_import="pinned",
        seed_snapshot_import_filter="pinned",
    )
    seed_run_dir = _write_seed_snapshot_run_fixture(
        tmp_path,
        updates=(10, 20, 30),
        champion_updates=(),
        pinned_policy_ids=("policy_000020",),
    )

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run_pinned_seed_filter",
    )

    expected_policy_id = result.train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000020",
    )
    registry = SnapshotRegistry.load(result.training_paths.snapshots_dir / "registry.json")
    assert result.imported_policy_ids == [expected_policy_id]
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [expected_policy_id]
    assert registry.champion_snapshots == [expected_policy_id]


def test_import_seed_snapshot_pool_can_use_explicit_registry_json(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    seed_run_dir = _write_seed_snapshot_run_fixture(tmp_path, updates=(10, 20, 30), champion_updates=(20,))
    source_registry = SnapshotRegistry.load(seed_run_dir / "training" / "snapshots" / "registry.json")
    source_registry.champion_snapshots = ["policy_000030"]
    explicit_registry = seed_run_dir / "training" / "snapshots" / "registry_explicit_champions.json"
    source_registry.save(explicit_registry)
    stack = _stack_with_seed_pool(
        tmp_path,
        seed_snapshot_import_filter="source_champions",
        seed_snapshot_registry_json=explicit_registry.as_posix(),
    )

    result = _import_seed_snapshot_pool_for_test(
        tmp_path,
        stack=stack,
        seed_run_dir=seed_run_dir,
        consumer_name="consumer_run_explicit_registry",
    )

    expected_policy_id = result.train_script._seed_snapshot_policy_id(
        source_run_dir=seed_run_dir.resolve(),
        source_policy_id="policy_000030",
    )
    registry = SnapshotRegistry.load(result.training_paths.snapshots_dir / "registry.json")
    assert result.imported_policy_ids == [expected_policy_id]
    assert [snapshot.policy_id for snapshot in registry.snapshots] == [expected_policy_id]
    assert registry.champion_snapshots == [expected_policy_id]


def _stack_with_seed_pool(tmp_path: Path, **pool_overrides: object):
    del tmp_path
    stack = load_stack_config(canonical_stack_config_path())
    assert stack.config.league is not None
    pool = replace(stack.config.league.pool, **pool_overrides)
    league = replace(stack.config.league, pool=pool)
    return replace(stack, config=replace(stack.config, league=league))
