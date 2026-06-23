from __future__ import annotations

from types import SimpleNamespace

from weiss_rl.runtime.components.opponents import (
    apply_opponent_pool_diversity_floor,
    filter_timeout_heavy_opponents,
    promotion_gated_recent_reservoir_size,
)


def test_promotion_recent_reservoir_matches_champion_gate_rules() -> None:
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=0,
            champion_size=8,
            admitted_champion_ids=("champion",),
            min_recent_size=2,
        )
        == 0
    )
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=8,
            champion_size=4,
            admitted_champion_ids=(),
            min_recent_size=2,
        )
        == 4
    )
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=8,
            champion_size=4,
            admitted_champion_ids=("champion",),
            min_recent_size=2,
        )
        == 2
    )


def test_filter_timeout_heavy_opponents_and_diversity_floor_preserve_pool_safety() -> None:
    assert filter_timeout_heavy_opponents(
        candidate_ids=("a",),
        league_config=SimpleNamespace(promotion_gate_enabled=False),
        outcomes=None,
        min_samples=32,
    ) == ("a",)

    league_config = SimpleNamespace(
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    outcomes = SimpleNamespace(
        counts=lambda policy_id: {
            "timeout_heavy": (0, 0, 0, 40),
            "healthy": (40, 0, 0, 0),
            "too_few": (0, 0, 0, 2),
        }[policy_id]
    )

    filtered = filter_timeout_heavy_opponents(
        candidate_ids=("timeout_heavy", "healthy", "too_few"),
        league_config=league_config,
        outcomes=outcomes,
        min_samples=32,
    )

    assert filtered == ("healthy", "too_few")
    assert apply_opponent_pool_diversity_floor(
        candidate_ids=("timeout_heavy", "healthy", "too_few"),
        filtered_candidate_ids=filtered[:1],
        minimum_floor_size=2,
    ) == (("healthy", "timeout_heavy"), 2)
    assert apply_opponent_pool_diversity_floor(
        candidate_ids=("timeout_heavy", "healthy"),
        filtered_candidate_ids=(),
        minimum_floor_size=2,
    ) == (("timeout_heavy", "healthy"), 0)
