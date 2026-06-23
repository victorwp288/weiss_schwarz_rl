from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.league.registry import (
    ChampionDemotion,
    SnapshotReferenceNormalization,
    SnapshotRegistry,
    champion_demotion_newer_than,
    champion_demotion_stale_by_age,
    normalize_snapshot_references,
    snapshot_weights_relpath,
)


def test_snapshot_registry_survives_restart_and_returns_latest_n(tmp_path: Path) -> None:
    registry_path = tmp_path / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry(recent_size=3, champion_size=1)

    registry.add_snapshot(
        policy_id="policy_000003",
        update=3,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000003"),
    )
    registry.add_snapshot(
        policy_id="policy_000001",
        update=1,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("policy_000001"),
    )
    registry.add_snapshot(
        policy_id="policy_000002",
        update=2,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000002"),
    )
    registry.save(registry_path)

    reloaded = SnapshotRegistry.load(registry_path)

    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [
        "policy_000001",
        "policy_000002",
        "policy_000003",
    ]
    assert reloaded.latest_ids(2) == ["policy_000002", "policy_000003"]


def test_snapshot_registry_prune_keeps_recent_champion_and_pinned_union() -> None:
    registry = SnapshotRegistry(recent_size=2, champion_size=1)
    for update in range(1, 6):
        policy_id = f"policy_{update:06d}"
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=(str(update) * 64)[:64],
            path=snapshot_weights_relpath(policy_id),
        )

    registry.pin_snapshot("policy_000002")
    registry.add_champion("policy_000003")
    registry.add_champion("policy_000004")

    pruned = registry.prune()

    assert [snapshot.policy_id for snapshot in registry.snapshots] == [
        "policy_000002",
        "policy_000004",
        "policy_000005",
    ]
    assert registry.champion_snapshots == ["policy_000004"]
    assert registry.pinned_snapshots == ["policy_000002"]
    assert [snapshot.policy_id for snapshot in pruned] == ["policy_000001", "policy_000003"]


def test_snapshot_registry_add_champion_dedupes_and_trims_window() -> None:
    registry = SnapshotRegistry(recent_size=0, champion_size=2)
    for update in range(1, 4):
        policy_id = f"policy_{update:06d}"
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=(str(update) * 64)[:64],
            path=snapshot_weights_relpath(policy_id),
        )

    registry.add_champion("policy_000001")
    registry.add_champion("policy_000002")
    registry.add_champion("policy_000001")
    registry.add_champion("policy_000003")

    assert registry.champion_snapshots == ["policy_000001", "policy_000003"]


def test_snapshot_reference_normalization_reports_dropped_and_trimmed_refs() -> None:
    result = normalize_snapshot_references(
        ["ghost", "policy_000001", "policy_000002", "policy_000001", "policy_000003"],
        existing_snapshot_ids={"policy_000001", "policy_000002", "policy_000003"},
        limit=2,
    )

    assert result == SnapshotReferenceNormalization(
        refs=["policy_000001", "policy_000003"],
        dropped_refs=["ghost"],
        trimmed_refs=["policy_000002"],
    )


def test_champion_demotion_helpers_preserve_order_and_report_remaining_refs() -> None:
    refs = ["policy_000080", "policy_000120", "policy_000160"]
    updates_by_policy = {
        "policy_000080": 80,
        "policy_000120": 120,
        "policy_000160": 160,
    }

    newer = champion_demotion_newer_than(
        refs,
        updates_by_policy=updates_by_policy,
        update=120,
    )
    stale = champion_demotion_stale_by_age(
        refs,
        updates_by_policy=updates_by_policy,
        current_update=180,
        max_age_updates=80,
    )

    assert newer == ChampionDemotion(
        removed_refs=["policy_000160"],
        remaining_refs=["policy_000080", "policy_000120"],
    )
    assert stale == ChampionDemotion(
        removed_refs=["policy_000080"],
        remaining_refs=["policy_000120", "policy_000160"],
    )


def test_snapshot_registry_add_champion_rejects_unknown_snapshot() -> None:
    registry = SnapshotRegistry()

    with pytest.raises(ValueError, match="existing snapshot"):
        registry.add_champion("policy_999999")


def test_snapshot_registry_load_normalizes_orphaned_refs(tmp_path: Path) -> None:
    registry_path = tmp_path / "training" / "snapshots" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 3,
                "champion_size": 2,
                "snapshots": [
                    {
                        "policy_id": "policy_000001",
                        "update": 1,
                        "weights_sha256": "a" * 64,
                        "path": snapshot_weights_relpath("policy_000001"),
                        "created_utc": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "policy_id": "policy_000002",
                        "update": 2,
                        "weights_sha256": "b" * 64,
                        "path": snapshot_weights_relpath("policy_000002"),
                        "created_utc": "2026-01-01T00:00:01+00:00",
                    },
                ],
                "champion_snapshots": ["ghost", "policy_000001", "policy_000001", "policy_000002"],
                "pinned_snapshots": ["ghost", "policy_000001", "policy_000001"],
            }
        ),
        encoding="utf-8",
    )

    registry = SnapshotRegistry.load(registry_path)

    assert registry.champion_snapshots == ["policy_000001", "policy_000002"]
    assert registry.pinned_snapshots == ["policy_000001"]


def test_snapshot_registry_rejects_checkpoint_paths() -> None:
    registry = SnapshotRegistry()

    try:
        registry.add_snapshot(
            policy_id="policy_000001",
            update=1,
            weights_sha256="a" * 64,
            path="training/checkpoints/checkpoint_1.pt",
        )
    except ValueError as exc:
        assert "training/snapshots" in str(exc)
    else:
        raise AssertionError("expected add_snapshot() to reject checkpoint paths")
