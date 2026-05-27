from __future__ import annotations

import numpy.testing as npt
import torch

from weiss_rl.models.public_heuristics import (
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    apply_public_heuristic_bias,
    attack_public_heuristic_raw,
    combine_public_heuristic_scores,
    default_public_heuristic_raw,
    hand_public_heuristic_raw,
    index_public_heuristic_raw,
    move_public_heuristic_raw,
    play_public_heuristic_raw,
    public_attack_profile,
    public_heuristic_slot_preference_array,
    public_prefer_lower,
    public_slot_action_score,
    slot_family_public_heuristic_raw,
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


def test_play_public_heuristic_raw_rewards_open_preferred_stage_slots() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    stage_slots = torch.tensor([0, 3, 5], dtype=torch.long)
    target_numeric = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float32,
    )

    raw = play_public_heuristic_raw(stage_slots, target_numeric, preferences, dtype=torch.float32)

    assert torch.equal(raw, torch.tensor([710.0, -1000.0, 650.0]))


def test_move_public_heuristic_raw_preserves_slot_improvement_and_validity_rules() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    from_slots = torch.tensor([3, 0, 2], dtype=torch.long)
    to_slots = torch.tensor([1, 4, 1], dtype=torch.long)
    source_numeric = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
    target_numeric = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)

    raw = move_public_heuristic_raw(
        from_slots,
        to_slots,
        source_numeric,
        target_numeric,
        preferences,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([187.0, -1000.0, -1000.0]))


def test_attack_public_heuristic_raw_preserves_attack_type_power_and_occupancy_rules() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    slot_values = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    attack_types = torch.tensor([2, 0, 1, 1], dtype=torch.long)
    source_numeric = torch.zeros((4, 7), dtype=torch.float32)
    source_numeric[:, 0] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    source_numeric[:, 3] = torch.tensor([1.0, 3.0, 2.0, 5.0])
    source_numeric[:, 5] = torch.tensor([2.0, 1.0, 1.0, 1.0])
    source_numeric[:, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    defender_numeric = torch.zeros((4, 7), dtype=torch.float32)
    defender_numeric[:, 0] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    defender_numeric[:, 3] = torch.tensor([0.0, 2.0, 5.0, 1.0])

    raw = attack_public_heuristic_raw(
        slot_values,
        attack_types,
        source_numeric,
        defender_numeric,
        preferences,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([1032.0, 1051.0, 1011.0, -1000.0]))


def test_attack_public_heuristic_raw_uses_catalog_attack_type_ids() -> None:
    preferences = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    slot_values = torch.tensor([0, 1, 2], dtype=torch.long)
    attack_types = torch.tensor([11, 7, 5], dtype=torch.long)
    source_numeric = torch.zeros((3, 7), dtype=torch.float32)
    source_numeric[:, 0] = 1.0
    source_numeric[:, 3] = torch.tensor([1.0, 3.0, 2.0])
    source_numeric[:, 5] = 0.0
    source_numeric[:, 6] = torch.tensor([0.0, 0.0, 1.0])
    defender_numeric = torch.zeros((3, 7), dtype=torch.float32)
    defender_numeric[:, 0] = torch.tensor([0.0, 1.0, 1.0])
    defender_numeric[:, 3] = torch.tensor([0.0, 2.0, 5.0])

    raw = attack_public_heuristic_raw(
        slot_values,
        attack_types,
        source_numeric,
        defender_numeric,
        preferences,
        direct_attack_type_id=11,
        frontal_attack_type_id=7,
        side_attack_type_id=5,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([980.0, 1005.0, 980.0]))


def test_slot_family_public_heuristic_raw_scores_encore_families_only() -> None:
    preferences = torch.tensor([20.0, 30.0, 15.0, 8.0, 6.0], dtype=torch.float32)
    family_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    slot_values = torch.tensor([0, 1, 4], dtype=torch.long)
    slot_numeric = torch.zeros((3, 7), dtype=torch.float32)
    slot_numeric[:, 3] = torch.tensor([1.0, 2.0, 3.0])

    raw = slot_family_public_heuristic_raw(
        family_ids,
        slot_values,
        slot_numeric,
        preferences,
        encore_pay_family_id=10,
        encore_decline_family_id=11,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([740.0, 180.0, 0.0]))


def test_hand_public_heuristic_raw_scores_family_specific_tactical_context() -> None:
    family_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    hand_indices = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long)

    raw = hand_public_heuristic_raw(
        family_ids,
        hand_indices,
        attackers_available=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0]),
        front_defenders=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        self_level_count=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        self_clock_count=torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0]),
        climax_play_family_id=1,
        clock_from_hand_family_id=2,
        main_event_family_id=3,
        mulligan_select_family_id=4,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([584.0, 533.0, 329.0, 117.0, 0.0]))


def test_index_public_heuristic_raw_scores_selection_and_paging_families() -> None:
    family_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    index_values = torch.tensor([0, 2, 3, 4, 5, 6], dtype=torch.long)
    choice_page_start = torch.tensor([0.0, 0.0, 0.0, 16.0, 12.0, 0.0])
    choice_total = torch.tensor([0.0, 0.0, 0.0, 40.0, 40.0, 0.0])

    raw = index_public_heuristic_raw(
        family_ids,
        index_values,
        choice_page_start=choice_page_start,
        choice_total=choice_total,
        choice_select_family_id=1,
        level_up_family_id=2,
        trigger_order_family_id=3,
        next_page_family_id=4,
        prev_page_family_id=5,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([300.0, 288.0, 277.0, 178.0, 182.0, 0.0]))


def test_default_public_heuristic_raw_scores_confirm_and_pass_families() -> None:
    family_ids = torch.tensor([7, 8, 9], dtype=torch.long)

    raw = default_public_heuristic_raw(
        family_ids,
        mulligan_confirm_family_id=7,
        pass_family_id=8,
        dtype=torch.float32,
    )

    assert torch.equal(raw, torch.tensor([260.0, 160.0, 0.0]))


def test_combine_public_heuristic_scores_preserves_weighting() -> None:
    combined = combine_public_heuristic_scores(
        torch.tensor([1.0, 2.0]),
        torch.tensor([4.0, 8.0]),
        torch.tensor([8.0, 4.0]),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([38.0, 73.0]))


def test_apply_public_heuristic_bias_supports_scale_and_family_allow_list() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    raw_scores = torch.tensor([100.0, 50.0, -20.0])

    gated = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=10.0,
        family_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        bias_family_ids=torch.tensor([2, 4], dtype=torch.long),
    )
    ungated = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=10.0,
        family_ids=None,
        bias_family_ids=torch.empty((0,), dtype=torch.long),
    )
    disabled = apply_public_heuristic_bias(
        scores,
        raw_scores,
        scale=0.0,
        family_ids=torch.tensor([2, 3, 4], dtype=torch.long),
        bias_family_ids=torch.tensor([2, 4], dtype=torch.long),
    )

    assert torch.equal(gated, torch.tensor([11.0, 2.0, 1.0]))
    assert torch.equal(ungated, torch.tensor([11.0, 7.0, 1.0]))
    assert torch.equal(disabled, scores)


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
