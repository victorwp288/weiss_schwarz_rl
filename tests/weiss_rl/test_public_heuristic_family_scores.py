from __future__ import annotations

import torch
from weiss_rl.models.public_heuristics import (
    default_public_heuristic_raw,
    hand_public_heuristic_raw,
    index_public_heuristic_raw,
    slot_family_public_heuristic_raw,
)


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
