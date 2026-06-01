from __future__ import annotations

import numpy as np

from weiss_rl.league.opponent_pool import (
    OpponentPoolSampler,
    compose_runtime_opponent_pool,
    opponent_sampling_distribution,
    resolve_opponent_win_rates,
    sample_opponent_snapshot_ids,
    select_opponent_snapshot_ids,
    select_opponent_snapshots,
    select_runtime_opponent_snapshots,
)
from weiss_rl.league.pfsp import pfsp_probabilities
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath


def _build_registry(snapshot_ids: list[str], *, champion_snapshot_ids: list[str]) -> SnapshotRegistry:
    registry = SnapshotRegistry()
    for update, snapshot_id in enumerate(snapshot_ids, start=1):
        registry.add_snapshot(
            policy_id=snapshot_id,
            update=update,
            weights_sha256=(snapshot_id * 64)[:64].ljust(64, "0"),
            path=snapshot_weights_relpath(snapshot_id),
        )
    for snapshot_id in champion_snapshot_ids:
        registry.add_champion(snapshot_id)
    return registry


def test_select_opponent_snapshot_ids_dedupes_recent_and_champion_overlap() -> None:
    registry = _build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s3"])

    selection = select_opponent_snapshots(registry, recent_size=2, champion_size=2)

    assert selection.recent_ids == ("s3", "s4")
    assert selection.champion_ids == ("s1", "s3")
    assert selection.candidate_ids == ("s3", "s4", "s1")
    assert select_opponent_snapshot_ids(registry, recent_size=2, champion_size=2) == selection.candidate_ids


def test_select_runtime_opponent_snapshots_preserves_runtime_lane_order_and_exclusions() -> None:
    registry = _build_registry(
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


def test_compose_runtime_opponent_pool_keeps_hard_negatives_first_and_removes_lane_overlap() -> None:
    registry = _build_registry(["champion", "recent_a", "recent_b"], champion_snapshot_ids=["champion"])
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
    registry = _build_registry(["champion", "recent"], champion_snapshot_ids=["champion"])
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


def test_select_opponent_snapshot_ids_uses_pruned_registry_state() -> None:
    registry = _build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s3"])
    registry.recent_size = 2
    registry.champion_size = 1
    registry.pin_snapshot("s2")
    registry.prune()

    snapshot_ids = select_opponent_snapshot_ids(registry, recent_size=2, champion_size=1)

    assert [snapshot.policy_id for snapshot in registry.snapshots] == ["s2", "s3", "s4"]
    assert snapshot_ids == ("s3", "s4")


def test_resolve_opponent_win_rates_uses_neutral_fallback_for_missing_stats() -> None:
    win_rates = resolve_opponent_win_rates(
        ("recent", "champion", "missing"),
        win_rates_by_snapshot_id={"recent": 0.8, "champion": 0.2},
    )

    assert np.array_equal(win_rates, np.array([0.8, 0.2, 0.5], dtype=np.float64))


def test_sample_opponent_snapshot_ids_matches_pfsp_distribution_deterministically() -> None:
    snapshot_ids = ("recent", "champion", "neutral")
    win_rates = {"recent": 0.8, "champion": 0.2}
    rng = np.random.default_rng(17)

    sampled = sample_opponent_snapshot_ids(
        snapshot_ids,
        count=6,
        rng=rng,
        win_rates_by_snapshot_id=win_rates,
        power=2.0,
        eps_uniform=0.2,
    )

    expected_probabilities = pfsp_probabilities(
        np.array([0.8, 0.2, 0.5], dtype=np.float64),
        power=2.0,
        eps_uniform=0.2,
    )
    expected_rng = np.random.default_rng(17)
    expected_indices = expected_rng.choice(len(snapshot_ids), size=6, replace=True, p=expected_probabilities)
    expected = tuple(snapshot_ids[index] for index in expected_indices.tolist())

    assert sampled == expected


def test_sample_opponent_snapshot_ids_applies_weight_multipliers_deterministically() -> None:
    snapshot_ids = ("old_hard", "new_hard", "focused_hard")
    win_rates = {"old_hard": 0.5, "new_hard": 0.5, "focused_hard": 0.5}
    multipliers = {"focused_hard": 5.0}

    sampled = sample_opponent_snapshot_ids(
        snapshot_ids,
        count=8,
        rng=np.random.default_rng(17),
        win_rates_by_snapshot_id=win_rates,
        weight_multipliers_by_snapshot_id=multipliers,
        power=2.0,
        eps_uniform=0.0,
    )

    expected_probabilities = pfsp_probabilities(
        np.array([0.5, 0.5, 0.5], dtype=np.float64),
        power=2.0,
        eps_uniform=0.0,
    )
    expected_probabilities = expected_probabilities * np.array([1.0, 1.0, 5.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)
    expected_rng = np.random.default_rng(17)
    expected_indices = expected_rng.choice(len(snapshot_ids), size=8, replace=True, p=expected_probabilities)
    expected = tuple(snapshot_ids[index] for index in expected_indices.tolist())

    assert sampled == expected


def test_opponent_sampling_distribution_exposes_resolved_win_rates_and_probabilities() -> None:
    distribution = opponent_sampling_distribution(
        ("old_hard", "new_hard", "focused_hard"),
        win_rates_by_snapshot_id={"old_hard": 0.5, "new_hard": 0.5, "focused_hard": 0.5},
        weight_multipliers_by_snapshot_id={"focused_hard": 5.0},
        power=2.0,
        eps_uniform=0.0,
    )

    expected_probabilities = pfsp_probabilities(
        np.array([0.5, 0.5, 0.5], dtype=np.float64),
        power=2.0,
        eps_uniform=0.0,
    )
    expected_probabilities = expected_probabilities * np.array([1.0, 1.0, 5.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)

    assert np.array_equal(distribution.win_rates, np.array([0.5, 0.5, 0.5], dtype=np.float64))
    assert np.array_equal(distribution.probabilities, expected_probabilities)


def test_opponent_pool_sampler_samples_from_recent_plus_champion_pool() -> None:
    registry = _build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s2"])
    sampler = OpponentPoolSampler(
        registry=registry,
        recent_size=2,
        champion_size=2,
        win_rates_by_snapshot_id={"s3": 0.8, "s4": 0.3, "s1": 0.1, "s2": 0.6},
    )

    rng = np.random.default_rng(23)
    sampled = sampler.sample(count=4, rng=rng)

    assert sampler.snapshot_ids() == ("s3", "s4", "s1", "s2")
    assert len(sampled) == 4
    assert set(sampled).issubset({"s1", "s2", "s3", "s4"})


def test_opponent_pool_sampler_empty_override_uses_neutral_fallback() -> None:
    registry = _build_registry(["s1", "s2"], champion_snapshot_ids=[])
    sampler = OpponentPoolSampler(
        registry=registry,
        recent_size=2,
        champion_size=0,
        win_rates_by_snapshot_id={"s1": 0.9, "s2": 0.1},
    )
    snapshot_ids = sampler.snapshot_ids()

    sampled = sampler.sample(count=6, rng=np.random.default_rng(7), win_rates_by_snapshot_id={})

    expected_probabilities = pfsp_probabilities(
        np.array([0.5, 0.5], dtype=np.float64),
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
    )
    expected_rng = np.random.default_rng(7)
    expected_indices = expected_rng.choice(len(snapshot_ids), size=6, replace=True, p=expected_probabilities)
    expected = tuple(snapshot_ids[index] for index in expected_indices.tolist())

    assert sampled == expected


def test_opponent_pool_sampler_none_override_uses_stored_win_rates() -> None:
    registry = _build_registry(["s1", "s2"], champion_snapshot_ids=[])
    sampler = OpponentPoolSampler(
        registry=registry,
        recent_size=2,
        champion_size=0,
        win_rates_by_snapshot_id={"s1": 0.9, "s2": 0.1},
    )
    snapshot_ids = sampler.snapshot_ids()

    sampled = sampler.sample(count=6, rng=np.random.default_rng(7), win_rates_by_snapshot_id=None)

    expected_probabilities = pfsp_probabilities(
        np.array([0.9, 0.1], dtype=np.float64),
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
    )
    expected_rng = np.random.default_rng(7)
    expected_indices = expected_rng.choice(len(snapshot_ids), size=6, replace=True, p=expected_probabilities)
    expected = tuple(snapshot_ids[index] for index in expected_indices.tolist())

    assert sampled == expected
