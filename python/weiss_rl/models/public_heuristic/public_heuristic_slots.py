"""Slot preferences and board-count helpers for public heuristic scoring."""

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


__all__ = [
    "PUBLIC_HEURISTIC_BACK_ROW_SLOTS",
    "PUBLIC_HEURISTIC_CENTER_SLOT",
    "PUBLIC_HEURISTIC_FRONT_ROW_SLOTS",
    "PUBLIC_HEURISTIC_SLOT_PREFERENCE",
    "public_attack_profile",
    "public_heuristic_slot_preference_array",
    "public_prefer_lower",
    "public_slot_action_score",
    "slot_preference_values",
]
