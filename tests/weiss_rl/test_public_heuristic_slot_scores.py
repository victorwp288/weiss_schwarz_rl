from __future__ import annotations

import numpy.testing as npt
import torch
from weiss_rl.models.public_heuristics import (
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    public_attack_profile,
    public_heuristic_slot_preference_array,
    public_prefer_lower,
    public_slot_action_score,
    slot_preference_values,
)


def test_public_heuristic_slot_preference_array_uses_known_slot_scores() -> None:
    preferences = public_heuristic_slot_preference_array(7)

    npt.assert_array_equal(preferences, [20.0, 30.0, 15.0, 8.0, 6.0, 0.0, 0.0])


def test_slot_preference_values_zeroes_invalid_slots_and_empty_preferences() -> None:
    preferences = torch.tensor([5.0, 7.0, 11.0], dtype=torch.float32)
    slot_indices = torch.tensor([0, 2, -1, 3], dtype=torch.long)

    values = slot_preference_values(slot_indices, preferences, dtype=torch.float64)

    assert values.dtype == torch.float64
    assert torch.equal(values, torch.tensor([5.0, 11.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(
        slot_preference_values(slot_indices, torch.empty((0,), dtype=torch.float32), dtype=torch.float32),
        torch.zeros((4,), dtype=torch.float32),
    )


def test_public_prefer_lower_scores_non_negative_values_only() -> None:
    values = torch.tensor([0, 2, -1, 5], dtype=torch.long)

    assert torch.equal(public_prefer_lower(values, dtype=torch.float32), torch.tensor([0.0, -2.0, 0.0, -5.0]))


def test_public_slot_action_score_combines_preferences_and_power_buckets() -> None:
    preferences = torch.tensor([10.0, 20.0], dtype=torch.float32)
    slot_values = torch.tensor([1, 0, 4], dtype=torch.long)
    slot_numeric = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.15],
            [0.0, 0.0, 0.0, -0.25],
            [0.0, 0.0, 0.0, 0.05],
        ],
        dtype=torch.float32,
    )

    scores = public_slot_action_score(slot_values, slot_numeric, preferences, dtype=torch.float32)

    assert torch.equal(scores, torch.tensor([23.0, 10.0, 1.0]))


def test_public_attack_profile_counts_ready_attackers_and_occupied_front_defenders() -> None:
    self_stage_numeric = torch.zeros((2, 5, 7), dtype=torch.float32)
    opponent_stage_numeric = torch.zeros((2, 5, 7), dtype=torch.float32)
    self_stage_numeric[0, 0, 0] = 1.0
    self_stage_numeric[0, 1, 0] = 1.0
    self_stage_numeric[0, 1, 2] = 1.0
    self_stage_numeric[0, 3, 0] = 1.0
    self_stage_numeric[1, 2, 0] = 1.0
    opponent_stage_numeric[0, 0, 0] = 1.0
    opponent_stage_numeric[0, 2, 0] = 1.0
    opponent_stage_numeric[1, 4, 0] = 1.0

    attackers, defenders = public_attack_profile(
        self_stage_numeric,
        opponent_stage_numeric,
        front_row_count=len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS),
        dtype=torch.float64,
    )

    assert attackers.dtype == torch.float64
    assert defenders.dtype == torch.float64
    assert torch.equal(attackers, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.equal(defenders, torch.tensor([2.0, 0.0], dtype=torch.float64))
