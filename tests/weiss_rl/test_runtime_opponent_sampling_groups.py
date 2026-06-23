from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.runtime.components.opponents import (
    RuntimeOpponentGroup,
    build_runtime_opponent_sampling_groups,
    build_runtime_opponent_sampling_plan,
    sample_runtime_opponent_group_policy_ids,
)

from .runtime_opponent_sampling_test_support import OpponentSamplingOutcomes


def test_build_runtime_opponent_sampling_groups_preserves_pre_pfsp_lane_order_and_mirror_remainder() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.9,
            hard_negative_mix_fraction=0.9,
        )
    )

    groups = build_runtime_opponent_sampling_groups(
        league_config=league_config,
        pfsp_ready=False,
        reference_update=0,
        mirror_weight=0.9,
        heuristic_public_weight=0.2,
        heuristic_public_variant_weight=0.2,
        noleague_baseline_weight=0.2,
        warmup_snapshot_weight=0.2,
        opponent_candidate_ids=("warm_a", "warm_b"),
        opponent_hard_negative_ids=("hard_a",),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a",),
        opponent_heuristic_policy_ids=("heuristic", "aggro", "control"),
        opponent_model_ids=("baseline", "warm_a", "warm_b"),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control", "missing"),
        noleague_baseline_policy_id="baseline",
    )

    assert [(group.name, group.policy_ids, group.weight) for group in groups] == [
        ("heuristic_public", ("heuristic",), 0.2),
        ("heuristic_public_variant", ("aggro", "control"), 0.2),
        ("noleague_baseline", ("baseline",), 0.2),
        ("warmup_snapshot", ("warm_a", "warm_b"), 0.2),
        ("mirror", ("mirror",), pytest.approx(0.2)),
    ]


def test_build_runtime_opponent_sampling_groups_preserves_pfsp_ready_lane_order_and_recent_remainder() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.25,
            hard_negative_mix_fraction=0.15,
        )
    )

    groups = build_runtime_opponent_sampling_groups(
        league_config=league_config,
        pfsp_ready=True,
        reference_update=0,
        mirror_weight=0.2,
        heuristic_public_weight=0.1,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.1,
        warmup_snapshot_weight=0.3,
        opponent_candidate_ids=("champ_a", "hard_a", "recent_a"),
        opponent_hard_negative_ids=("hard_a",),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a",),
        opponent_heuristic_policy_ids=("heuristic",),
        opponent_model_ids=("baseline", "champ_a", "hard_a", "recent_a"),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=(),
        noleague_baseline_policy_id="baseline",
    )

    assert [(group.name, group.policy_ids, group.weight) for group in groups] == [
        ("heuristic_public", ("heuristic",), 0.1),
        ("noleague_baseline", ("baseline",), 0.1),
        ("mirror", ("mirror",), 0.2),
        ("hard_negative", ("hard_a",), 0.15),
        ("champion", ("champ_a",), 0.25),
        ("recent", ("recent_a",), pytest.approx(0.2)),
    ]


def test_runtime_opponent_sampling_plan_preserves_zero_weight_uniform_fallback() -> None:
    plan = build_runtime_opponent_sampling_plan(
        (
            RuntimeOpponentGroup(name="hard_negative", policy_ids=("hard_a",), weight=0.0),
            RuntimeOpponentGroup(name="champion", policy_ids=("champ_a",), weight=0.0),
        )
    )

    assert [group.name for group in plan.groups] == ["hard_negative", "champion"]
    assert np.array_equal(plan.probabilities, np.asarray([0.5, 0.5], dtype=np.float64))


def test_sample_runtime_opponent_group_policy_ids_preserves_variant_rng_draws() -> None:
    group = RuntimeOpponentGroup(name="heuristic_public_variant", policy_ids=("aggro", "control"), weight=1.0)

    sampled = sample_runtime_opponent_group_policy_ids(
        group=group,
        count=5,
        rng=np.random.default_rng(3),
        league_config=SimpleNamespace(pfsp_power=2.0, pfsp_epsilon_uniform=0.0),
        outcomes=OpponentSamplingOutcomes(),
    )

    expected_indices = np.random.default_rng(3).integers(2, size=5)
    assert sampled == tuple(("aggro", "control")[int(index)] for index in expected_indices)
