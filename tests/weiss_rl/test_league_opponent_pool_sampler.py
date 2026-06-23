from __future__ import annotations

import numpy as np
from weiss_rl.league.opponent_pool import OpponentPoolSampler
from weiss_rl.league.pfsp import pfsp_probabilities

from .league_opponent_pool_test_support import build_registry


def test_opponent_pool_sampler_samples_from_recent_plus_champion_pool() -> None:
    registry = build_registry(["s1", "s2", "s3", "s4"], champion_snapshot_ids=["s1", "s2"])
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
    registry = build_registry(["s1", "s2"], champion_snapshot_ids=[])
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
    registry = build_registry(["s1", "s2"], champion_snapshot_ids=[])
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
