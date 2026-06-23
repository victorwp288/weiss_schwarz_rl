from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.training.snapshots import demote_registry_champions_newer_than


def test_demote_registry_champions_newer_than_removes_newer_refs_only(tmp_path: Path) -> None:
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000120")

    removed = registry.demote_champions_newer_than(80)

    assert removed == ["policy_000120"]
    assert registry.champion_snapshots == ["policy_000080"]


def test_demote_registry_champions_newer_than_updates_registry_file(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True)
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000120")
    registry.save(snapshots_dir / "registry.json")

    removed = demote_registry_champions_newer_than(
        SimpleNamespace(snapshots_dir=snapshots_dir),
        update_count=80,
    )

    assert removed == ["policy_000120"]
    reloaded = SnapshotRegistry.load(snapshots_dir / "registry.json")
    assert reloaded.champion_snapshots == ["policy_000080"]


def test_demote_stale_champions_removes_old_refs_only() -> None:
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000080",
        update=80,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000080/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000180",
        update=180,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000180/weights.pt",
    )
    registry.add_champion("policy_000080")
    registry.add_champion("policy_000180")

    removed = registry.demote_stale_champions(current_update=220, max_age_updates=60)

    assert removed == ["policy_000080"]
    assert registry.champion_snapshots == ["policy_000180"]
