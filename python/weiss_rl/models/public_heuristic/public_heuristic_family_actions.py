"""Family, hand-index, and generic-index public heuristic scoring rules."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.models.public_heuristic.public_heuristic_slots import public_prefer_lower, slot_preference_values


def slot_family_public_heuristic_raw(
    family_ids: Tensor,
    slot_values: Tensor,
    slot_numeric: Tensor,
    public_slot_preference: Tensor,
    *,
    encore_pay_family_id: int,
    encore_decline_family_id: int,
    dtype: torch.dtype,
) -> Tensor:
    """Score slot-family public heuristic actions such as encore decisions."""

    slot_pref = slot_preference_values(slot_values, public_slot_preference, dtype=dtype)
    power_term = slot_numeric[:, 3].to(dtype=dtype) * 20.0
    raw = slot_values.new_zeros(slot_values.shape, dtype=dtype)
    if int(encore_pay_family_id) >= 0:
        raw = torch.where(
            family_ids == int(encore_pay_family_id),
            slot_values.new_full(slot_values.shape, 700.0, dtype=dtype) + slot_pref + power_term,
            raw,
        )
    if int(encore_decline_family_id) >= 0:
        raw = torch.where(
            family_ids == int(encore_decline_family_id),
            slot_values.new_full(slot_values.shape, 110.0, dtype=dtype) + slot_pref + power_term,
            raw,
        )
    return raw


def hand_public_heuristic_raw(
    family_ids: Tensor,
    hand_indices: Tensor,
    *,
    attackers_available: Tensor,
    front_defenders: Tensor,
    self_level_count: Tensor,
    self_clock_count: Tensor,
    climax_play_family_id: int,
    clock_from_hand_family_id: int,
    main_event_family_id: int,
    mulligan_select_family_id: int,
    dtype: torch.dtype,
) -> Tensor:
    """Score hand-index public heuristic actions from tactical context."""

    raw = hand_indices.new_zeros(hand_indices.shape, dtype=dtype)
    lower_index_bonus = public_prefer_lower(hand_indices, dtype=dtype)
    if int(climax_play_family_id) >= 0:
        climax_bonus = (
            attackers_available * 10.0
            + front_defenders * 4.0
            + torch.where(
                attackers_available > 0.0,
                hand_indices.new_full(hand_indices.shape, 10.0, dtype=dtype),
                hand_indices.new_full(hand_indices.shape, -20.0, dtype=dtype),
            )
        )
        raw = torch.where(
            family_ids == int(climax_play_family_id),
            hand_indices.new_full(hand_indices.shape, 550.0, dtype=dtype) + climax_bonus + lower_index_bonus,
            raw,
        )
    if int(clock_from_hand_family_id) >= 0:
        clock_bonus = torch.where(
            (self_level_count <= 0.0) & (self_clock_count < 6.0),
            40.0 - self_clock_count,
            self_clock_count.new_full(self_clock_count.shape, 10.0, dtype=dtype),
        )
        raw = torch.where(
            family_ids == int(clock_from_hand_family_id),
            hand_indices.new_full(hand_indices.shape, 500.0, dtype=dtype) + clock_bonus + lower_index_bonus,
            raw,
        )
    if int(main_event_family_id) >= 0:
        raw = torch.where(
            family_ids == int(main_event_family_id),
            hand_indices.new_full(hand_indices.shape, 330.0, dtype=dtype) + lower_index_bonus,
            raw,
        )
    if int(mulligan_select_family_id) >= 0:
        raw = torch.where(
            family_ids == int(mulligan_select_family_id),
            hand_indices.new_full(hand_indices.shape, 120.0, dtype=dtype) + lower_index_bonus,
            raw,
        )
    return raw


def index_public_heuristic_raw(
    family_ids: Tensor,
    index_values: Tensor,
    *,
    choice_page_start: Tensor,
    choice_total: Tensor,
    choice_select_family_id: int,
    level_up_family_id: int,
    trigger_order_family_id: int,
    next_page_family_id: int,
    prev_page_family_id: int,
    dtype: torch.dtype,
) -> Tensor:
    """Score generic index public heuristic actions."""

    raw = index_values.new_zeros(index_values.shape, dtype=dtype)
    lower_index_bonus = public_prefer_lower(index_values, dtype=dtype)
    if int(choice_select_family_id) >= 0:
        raw = torch.where(
            family_ids == int(choice_select_family_id),
            index_values.new_full(index_values.shape, 300.0, dtype=dtype) + lower_index_bonus,
            raw,
        )
    if int(level_up_family_id) >= 0:
        raw = torch.where(
            family_ids == int(level_up_family_id),
            index_values.new_full(index_values.shape, 290.0, dtype=dtype) + lower_index_bonus,
            raw,
        )
    if int(trigger_order_family_id) >= 0:
        raw = torch.where(
            family_ids == int(trigger_order_family_id),
            index_values.new_full(index_values.shape, 280.0, dtype=dtype) + lower_index_bonus,
            raw,
        )
    if int(next_page_family_id) >= 0:
        raw = torch.where(
            family_ids == int(next_page_family_id),
            index_values.new_full(index_values.shape, 170.0, dtype=dtype)
            + torch.clamp(choice_total - (choice_page_start + 16.0), min=0.0),
            raw,
        )
    if int(prev_page_family_id) >= 0:
        raw = torch.where(
            family_ids == int(prev_page_family_id),
            index_values.new_full(index_values.shape, 170.0, dtype=dtype) + torch.clamp(choice_page_start, min=0.0),
            raw,
        )
    return raw


def default_public_heuristic_raw(
    family_ids: Tensor,
    *,
    mulligan_confirm_family_id: int,
    pass_family_id: int,
    dtype: torch.dtype,
) -> Tensor:
    """Score public heuristic default-family actions."""

    raw = family_ids.new_zeros(family_ids.shape, dtype=dtype)
    if int(mulligan_confirm_family_id) >= 0:
        raw = torch.where(
            family_ids == int(mulligan_confirm_family_id),
            family_ids.new_full(family_ids.shape, 260.0, dtype=dtype),
            raw,
        )
    if int(pass_family_id) >= 0:
        raw = torch.where(
            family_ids == int(pass_family_id),
            family_ids.new_full(family_ids.shape, 160.0, dtype=dtype),
            raw,
        )
    return raw


__all__ = [
    "default_public_heuristic_raw",
    "hand_public_heuristic_raw",
    "index_public_heuristic_raw",
    "slot_family_public_heuristic_raw",
]
