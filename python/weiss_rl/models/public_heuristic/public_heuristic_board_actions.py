"""Board-action public heuristic scoring rules."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.models.public_heuristic.public_heuristic_slots import (
    PUBLIC_HEURISTIC_BACK_ROW_SLOTS,
    PUBLIC_HEURISTIC_CENTER_SLOT,
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    slot_preference_values,
)


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


__all__ = [
    "attack_public_heuristic_raw",
    "move_public_heuristic_raw",
    "play_public_heuristic_raw",
]
