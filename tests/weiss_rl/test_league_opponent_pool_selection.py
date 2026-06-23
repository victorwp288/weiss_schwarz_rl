from __future__ import annotations

from weiss_rl.league.opponent_pool import (
    select_opponent_snapshot_ids,
    select_opponent_snapshots,
    select_runtime_opponent_snapshots,
)

from .league_opponent_pool_test_support import build_registry


def test_select_opponent_snapshot_ids_dedupes_recent_and_champion_overlap() -> None:
    registry = build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s3"])

    selection = select_opponent_snapshots(registry, recent_size=2, champion_size=2)

    assert selection.recent_ids == ("s3", "s4")
    assert selection.champion_ids == ("s1", "s3")
    assert selection.candidate_ids == ("s3", "s4", "s1")
    assert select_opponent_snapshot_ids(registry, recent_size=2, champion_size=2) == selection.candidate_ids


def test_select_runtime_opponent_snapshots_preserves_runtime_lane_order_and_exclusions() -> None:
    registry = build_registry(
        ["champion_old", "recent_new", "b1_noleague_baseline"],
        champion_snapshot_ids=["champion_old", "b1_noleague_baseline"],
    )

    selection = select_runtime_opponent_snapshots(
        registry,
        recent_size=2,
        champion_ids=("champion_old", "b1_noleague_baseline"),
        excluded_policy_ids=("b1_noleague_baseline",),
    )

    assert selection.champion_ids == ("champion_old",)
    assert selection.recent_ids == ("recent_new",)
    assert selection.candidate_ids == ("champion_old", "recent_new")


def test_select_opponent_snapshot_ids_uses_pruned_registry_state() -> None:
    registry = build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s3"])
    registry.recent_size = 2
    registry.champion_size = 1
    registry.pin_snapshot("s2")
    registry.prune()

    snapshot_ids = select_opponent_snapshot_ids(registry, recent_size=2, champion_size=1)

    assert [snapshot.policy_id for snapshot in registry.snapshots] == ["s2", "s3", "s4"]
    assert snapshot_ids == ("s3", "s4")
