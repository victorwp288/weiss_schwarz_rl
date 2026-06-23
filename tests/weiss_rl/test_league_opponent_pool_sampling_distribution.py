from __future__ import annotations

import numpy as np
from weiss_rl.league.opponent_pool import (
    opponent_sampling_distribution,
    resolve_opponent_win_rates,
    sample_opponent_snapshot_ids,
)
from weiss_rl.league.pfsp import pfsp_probabilities


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
