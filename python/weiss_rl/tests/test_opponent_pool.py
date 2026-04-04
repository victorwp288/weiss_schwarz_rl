from __future__ import annotations

import numpy as np

from weiss_rl.league.opponent_pool import (
    OpponentPoolSampler,
    resolve_opponent_win_rates,
    sample_opponent_snapshot_ids,
    select_opponent_snapshot_ids,
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

    snapshot_ids = select_opponent_snapshot_ids(registry, recent_size=2, champion_size=2)

    assert snapshot_ids == ("s3", "s4", "s1")


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
