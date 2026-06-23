from __future__ import annotations

from weiss_rl.league.opponent_pool import compose_runtime_opponent_pool, select_runtime_opponent_snapshots

from .league_opponent_pool_test_support import build_registry


def test_compose_runtime_opponent_pool_keeps_hard_negatives_first_and_removes_lane_overlap() -> None:
    registry = build_registry(["champion", "recent_a", "recent_b"], champion_snapshot_ids=["champion"])
    selection = select_runtime_opponent_snapshots(
        registry,
        recent_size=2,
        champion_ids=("champion",),
    )

    composition = compose_runtime_opponent_pool(
        selection=selection,
        candidate_ids=("champion", "recent_a", "recent_b"),
        hard_negative_ids=("recent_a", "champion"),
        hard_negative_overlaps_champions=False,
    )

    assert composition.hard_negative_ids == ("recent_a", "champion")
    assert composition.champion_ids == ()
    assert composition.recent_ids == ("recent_b",)
    assert composition.candidate_ids == ("recent_a", "champion", "recent_b")


def test_compose_runtime_opponent_pool_can_account_hard_negative_champions_in_both_lanes() -> None:
    registry = build_registry(["champion", "recent"], champion_snapshot_ids=["champion"])
    selection = select_runtime_opponent_snapshots(
        registry,
        recent_size=2,
        champion_ids=("champion",),
    )

    composition = compose_runtime_opponent_pool(
        selection=selection,
        candidate_ids=("champion", "recent"),
        hard_negative_ids=("champion",),
        hard_negative_overlaps_champions=True,
    )

    assert composition.hard_negative_ids == ("champion",)
    assert composition.champion_ids == ("champion",)
    assert composition.recent_ids == ("recent",)
    assert composition.candidate_ids == ("champion", "recent")
