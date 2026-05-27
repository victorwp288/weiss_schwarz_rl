"""Candidate family partition helpers for structured model scoring."""

from __future__ import annotations

import torch
from torch import Tensor


def partition_candidate_family_indices(
    family_ids: Tensor,
    *,
    play_character_family_id: int,
    hand_family_ids: tuple[int, ...],
    main_move_family_id: int,
    attack_family_id: int,
    slot_family_ids: tuple[int, ...],
    index_family_ids: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Partition candidate row indices into structured scoring groups."""

    device = family_ids.device
    play_mask = family_ids == play_character_family_id
    hand_mask = torch.zeros_like(play_mask)
    for family_id in hand_family_ids:
        hand_mask |= family_ids == family_id
    move_mask = family_ids == main_move_family_id
    attack_mask = family_ids == attack_family_id
    slot_mask = torch.zeros_like(play_mask)
    for family_id in slot_family_ids:
        slot_mask |= family_ids == family_id
    index_mask = torch.zeros_like(play_mask)
    for family_id in index_family_ids:
        index_mask |= family_ids == family_id
    default_mask = ~(play_mask | hand_mask | move_mask | attack_mask | slot_mask | index_mask)

    def _indices(mask: Tensor) -> Tensor:
        if not torch.any(mask):
            return torch.zeros((0,), device=device, dtype=torch.long)
        return torch.nonzero(mask, as_tuple=False).squeeze(1)

    return (
        _indices(play_mask),
        _indices(hand_mask),
        _indices(move_mask),
        _indices(attack_mask),
        _indices(slot_mask),
        _indices(index_mask),
        _indices(default_mask),
    )
