from __future__ import annotations

import numpy as np

from .runtime_opponent_sampling_call_test_support import sample_runtime_policy_ids, sampling_config


def test_sample_runtime_opponent_policy_ids_preserves_pfsp_ready_rng_order_and_counters() -> None:
    result = sample_runtime_policy_ids(
        count=12,
        rng_seed=11,
        league_config=sampling_config(champion_mix_fraction=0.25, hard_negative_mix_fraction=0.25),
        pfsp_ready=True,
        opponent_candidate_ids=("champ_a", "recent_a", "recent_b", "hard_a", "hard_b"),
        opponent_hard_negative_ids=("hard_a", "hard_b"),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a", "recent_b"),
        opponent_model_ids=("champ_a", "recent_a", "recent_b", "hard_a", "hard_b"),
    )

    assert result.policy_ids == (
        "hard_b",
        "champ_a",
        "recent_b",
        "hard_a",
        "hard_a",
        "recent_b",
        "hard_b",
        "hard_b",
        "recent_b",
        "recent_b",
        "champ_a",
        "recent_b",
    )
    assert result.sampled_envs == 12
    assert result.champion_envs == 2
    assert result.recent_envs == 5
    assert result.hard_negative_envs == 5
    assert result.mirror_envs == 0
    assert dict(result.sampled_policy_envs) == {
        "champ_a": 2,
        "hard_a": 2,
        "hard_b": 3,
        "recent_b": 5,
    }
    assert dict(result.champion_policy_envs) == {"champ_a": 2}
    assert dict(result.hard_negative_policy_envs) == {"hard_a": 2, "hard_b": 3}
    assert dict(result.recent_policy_envs) == {"recent_b": 5}


def test_sample_runtime_opponent_policy_ids_weights_focused_hard_negatives() -> None:
    result = sample_runtime_policy_ids(
        count=8,
        rng_seed=17,
        league_config=sampling_config(
            hard_negative_mix_fraction=1.0,
            hard_negative_focus_policy_ids=("hard_b",),
            hard_negative_focus_weight_multiplier=5.0,
        ),
        pfsp_ready=True,
        opponent_candidate_ids=("hard_a", "hard_b"),
        opponent_hard_negative_ids=("hard_a", "hard_b"),
        opponent_model_ids=("hard_a", "hard_b"),
    )

    base_probabilities = np.asarray([((1.0 - 0.2) ** 2.0), ((1.0 - 0.4) ** 2.0)], dtype=np.float64)
    base_probabilities = base_probabilities / np.sum(base_probabilities)
    expected_probabilities = base_probabilities * np.array([1.0, 5.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)
    expected_rng = np.random.default_rng(17)
    expected_rng.choice(1, size=8, replace=True, p=np.array([1.0], dtype=np.float64))
    expected_indices = expected_rng.choice(2, size=8, replace=True, p=expected_probabilities)
    expected = tuple(("hard_a", "hard_b")[index] for index in expected_indices.tolist())

    assert result.policy_ids == expected
    assert result.hard_negative_envs == 8
    assert sum(dict(result.hard_negative_policy_envs).values()) == 8


def test_sample_runtime_opponent_policy_ids_applies_row_deficit_weights_to_champions() -> None:
    result = sample_runtime_policy_ids(
        count=8,
        rng_seed=19,
        league_config=sampling_config(
            champion_mix_fraction=1.0,
            row_deficit_policy_weights=(("champ_b", 4.0),),
        ),
        pfsp_ready=True,
        opponent_candidate_ids=("champ_a", "champ_b"),
        opponent_champion_ids=("champ_a", "champ_b"),
        opponent_model_ids=("champ_a", "champ_b"),
    )

    base_probabilities = np.asarray([((1.0 - 0.5) ** 2.0), ((1.0 - 0.5) ** 2.0)], dtype=np.float64)
    base_probabilities = base_probabilities / np.sum(base_probabilities)
    expected_probabilities = base_probabilities * np.array([1.0, 4.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)
    expected_rng = np.random.default_rng(19)
    expected_rng.choice(1, size=8, replace=True, p=np.array([1.0], dtype=np.float64))
    expected_indices = expected_rng.choice(2, size=8, replace=True, p=expected_probabilities)
    expected = tuple(("champ_a", "champ_b")[index] for index in expected_indices.tolist())

    assert result.policy_ids == expected
    assert result.champion_envs == 8
