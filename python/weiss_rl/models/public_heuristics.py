"""Public-heuristic tensor helpers for structured model scoring."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

PUBLIC_HEURISTIC_FRONT_ROW_SLOTS = frozenset({0, 1, 2})
PUBLIC_HEURISTIC_BACK_ROW_SLOTS = frozenset({3, 4})
PUBLIC_HEURISTIC_CENTER_SLOT = 1
PUBLIC_HEURISTIC_SLOT_PREFERENCE = {
    0: 20.0,
    1: 30.0,
    2: 15.0,
    3: 8.0,
    4: 6.0,
}


def public_heuristic_slot_preference_array(stage_slot_count: int) -> np.ndarray:
    """Return the dense public-heuristic slot preference table for a stage layout."""

    slot_preference = np.zeros((int(stage_slot_count),), dtype=np.float32)
    for slot_index in range(int(stage_slot_count)):
        slot_preference[slot_index] = float(PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(slot_index, 0.0))
    return slot_preference


def slot_preference_values(slot_indices: Tensor, public_slot_preference: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Look up public slot preferences while zeroing invalid slot indices."""

    if public_slot_preference.numel() == 0:
        return slot_indices.new_zeros(slot_indices.shape, dtype=dtype)
    valid = (slot_indices >= 0) & (slot_indices < int(public_slot_preference.shape[0]))
    safe_slots = torch.where(valid, slot_indices, torch.zeros_like(slot_indices)).to(dtype=torch.long)
    values = public_slot_preference.index_select(0, safe_slots).to(dtype=dtype)
    return values * valid.to(dtype=dtype)


def public_prefer_lower(values: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Score non-negative indices by preferring lower values."""

    return torch.where(values >= 0, -values.to(dtype=dtype), values.new_zeros(values.shape, dtype=dtype))


def public_slot_action_score(
    slot_values: Tensor,
    slot_numeric: Tensor,
    public_slot_preference: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Score a slot action from slot preference and clamped power buckets."""

    power = torch.clamp(slot_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
    return slot_preference_values(slot_values, public_slot_preference, dtype=dtype) + torch.floor(power / 1000.0)


def play_public_heuristic_raw(
    stage_slots: Tensor,
    target_numeric: Tensor,
    public_slot_preference: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Score public play actions from target occupancy and stage-slot preference."""

    slot_pref = slot_preference_values(stage_slots, public_slot_preference, dtype=dtype)
    front_bonus = torch.where(
        stage_slots < len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS),
        stage_slots.new_full(stage_slots.shape, 40.0, dtype=dtype),
        torch.where(
            stage_slots < len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS) + len(PUBLIC_HEURISTIC_BACK_ROW_SLOTS),
            stage_slots.new_full(stage_slots.shape, 20.0, dtype=dtype),
            stage_slots.new_zeros(stage_slots.shape, dtype=dtype),
        ),
    )
    occupied = target_numeric[:, 0].to(dtype=dtype) > 0.5
    raw = stage_slots.new_full(stage_slots.shape, 650.0, dtype=dtype) + slot_pref + front_bonus
    return torch.where(occupied, stage_slots.new_full(stage_slots.shape, -1000.0, dtype=dtype), raw)


def move_public_heuristic_raw(
    from_slots: Tensor,
    to_slots: Tensor,
    source_numeric: Tensor,
    target_numeric: Tensor,
    public_slot_preference: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Score public move actions from source/target occupancy and slot improvement."""

    source_pref = slot_preference_values(from_slots, public_slot_preference, dtype=dtype)
    target_pref = slot_preference_values(to_slots, public_slot_preference, dtype=dtype)
    improvement = target_pref - source_pref
    front_row_threshold = len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS)
    back_to_front = (from_slots >= front_row_threshold) & (to_slots < front_row_threshold)
    move_to_center = (to_slots == PUBLIC_HEURISTIC_CENTER_SLOT) & (from_slots != PUBLIC_HEURISTIC_CENTER_SLOT)
    bonus = back_to_front.to(dtype=dtype) * 30.0 + move_to_center.to(dtype=dtype) * 15.0
    valid = (source_numeric[:, 0].to(dtype=dtype) > 0.5) & (target_numeric[:, 0].to(dtype=dtype) <= 0.5)
    raw = from_slots.new_full(from_slots.shape, 120.0, dtype=dtype) + improvement + bonus
    return torch.where(valid, raw, from_slots.new_full(from_slots.shape, -1000.0, dtype=dtype))


def attack_public_heuristic_raw(
    slot_values: Tensor,
    attack_type_values: Tensor,
    source_numeric: Tensor,
    defender_numeric: Tensor,
    public_slot_preference: Tensor,
    *,
    direct_attack_type_id: int = 2,
    frontal_attack_type_id: int = 0,
    side_attack_type_id: int = 1,
    dtype: torch.dtype,
) -> Tensor:
    """Score public attack actions from attacker state, defender state, and attack type."""

    slot_pref = slot_preference_values(slot_values, public_slot_preference, dtype=dtype)
    attacker_occupied = source_numeric[:, 0].to(dtype=dtype) > 0.5
    attacker_power = source_numeric[:, 3].to(dtype=dtype)
    attacker_effective_soul = source_numeric[:, 5].to(dtype=dtype)
    side_attack_allowed = source_numeric[:, 6].to(dtype=dtype) > 0.5
    defender_occupied = defender_numeric[:, 0].to(dtype=dtype) > 0.5
    defender_power = defender_numeric[:, 3].to(dtype=dtype)
    attack_type_score = slot_values.new_zeros(slot_values.shape, dtype=dtype)
    if int(direct_attack_type_id) >= 0:
        direct_mask = attack_type_values == int(direct_attack_type_id)
        attack_type_score = torch.where(
            direct_mask,
            torch.where(
                defender_occupied,
                slot_values.new_full(slot_values.shape, 15.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 60.0, dtype=dtype),
            ),
            attack_type_score,
        )
    if int(frontal_attack_type_id) >= 0:
        frontal_mask = attack_type_values == int(frontal_attack_type_id)
        attack_type_score = torch.where(
            frontal_mask,
            torch.where(
                attacker_power >= defender_power,
                slot_values.new_full(slot_values.shape, 45.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 25.0, dtype=dtype),
            ),
            attack_type_score,
        )
    if int(side_attack_type_id) >= 0:
        side_mask = attack_type_values == int(side_attack_type_id)
        attack_type_score = torch.where(
            side_mask,
            torch.where(
                side_attack_allowed,
                slot_values.new_full(slot_values.shape, 40.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 5.0, dtype=dtype),
            ),
            attack_type_score,
        )
    power_term = attacker_power * 20.0
    soul_term = attacker_effective_soul * 16.0
    raw = (
        slot_values.new_full(slot_values.shape, 900.0, dtype=dtype)
        + attack_type_score
        + slot_pref
        + power_term
        + soul_term
    )
    return torch.where(attacker_occupied, raw, slot_values.new_full(slot_values.shape, -1000.0, dtype=dtype))


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


def combine_public_heuristic_scores(
    score0: Tensor,
    score1: Tensor,
    score2: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Combine the three packed public-heuristic score components."""

    return score0.to(dtype=dtype) * 32.0 + score1.to(dtype=dtype) + (score2.to(dtype=dtype) / 4.0)


def apply_public_heuristic_bias(
    scores: Tensor,
    raw_scores: Tensor,
    *,
    scale: float,
    family_ids: Tensor | None,
    bias_family_ids: Tensor,
) -> Tensor:
    """Apply optional public-heuristic logit bias with an optional family allow-list."""

    if scale <= 0.0 or raw_scores.numel() == 0:
        return scores
    bias = raw_scores.to(dtype=scores.dtype) * (float(scale) / 100.0)
    if family_ids is None or bias_family_ids.numel() == 0:
        return scores + bias
    allowed = torch.isin(
        family_ids.to(device=bias_family_ids.device, dtype=torch.long),
        bias_family_ids,
    ).to(device=scores.device, dtype=scores.dtype)
    return scores + (bias * allowed)


def public_attack_profile(
    self_stage_numeric: Tensor,
    opponent_stage_numeric: Tensor,
    *,
    front_row_count: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Return available front-row attackers and occupied opposing front slots."""

    front_count = int(front_row_count)
    attackers_available = (
        (
            (self_stage_numeric[:, :front_count, 0].to(dtype=dtype) > 0.5)
            & ~(self_stage_numeric[:, :front_count, 2].to(dtype=dtype) > 0.5)
        )
        .sum(dim=1)
        .to(dtype=dtype)
    )
    front_defenders = (opponent_stage_numeric[:, :front_count, 0].to(dtype=dtype) > 0.5).sum(dim=1).to(dtype=dtype)
    return attackers_available, front_defenders
