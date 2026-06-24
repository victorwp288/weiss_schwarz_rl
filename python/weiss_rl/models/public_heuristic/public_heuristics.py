"""Compatibility facade for public heuristic model-scoring helpers."""

from __future__ import annotations

from weiss_rl.models.public_heuristic.public_heuristic_bias import (
    apply_public_heuristic_bias,
    combine_public_heuristic_scores,
)
from weiss_rl.models.public_heuristic.public_heuristic_board_actions import (
    attack_public_heuristic_raw,
    move_public_heuristic_raw,
    play_public_heuristic_raw,
)
from weiss_rl.models.public_heuristic.public_heuristic_family_actions import (
    default_public_heuristic_raw,
    hand_public_heuristic_raw,
    index_public_heuristic_raw,
    slot_family_public_heuristic_raw,
)
from weiss_rl.models.public_heuristic.public_heuristic_slots import (
    PUBLIC_HEURISTIC_BACK_ROW_SLOTS,
    PUBLIC_HEURISTIC_CENTER_SLOT,
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    PUBLIC_HEURISTIC_SLOT_PREFERENCE,
    public_attack_profile,
    public_heuristic_slot_preference_array,
    public_prefer_lower,
    public_slot_action_score,
    slot_preference_values,
)

__all__ = [
    "PUBLIC_HEURISTIC_BACK_ROW_SLOTS",
    "PUBLIC_HEURISTIC_CENTER_SLOT",
    "PUBLIC_HEURISTIC_FRONT_ROW_SLOTS",
    "PUBLIC_HEURISTIC_SLOT_PREFERENCE",
    "apply_public_heuristic_bias",
    "attack_public_heuristic_raw",
    "combine_public_heuristic_scores",
    "default_public_heuristic_raw",
    "hand_public_heuristic_raw",
    "index_public_heuristic_raw",
    "move_public_heuristic_raw",
    "play_public_heuristic_raw",
    "public_attack_profile",
    "public_heuristic_slot_preference_array",
    "public_prefer_lower",
    "public_slot_action_score",
    "slot_family_public_heuristic_raw",
    "slot_preference_values",
]
