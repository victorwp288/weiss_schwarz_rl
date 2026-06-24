"""Candidate family partition helpers for structured model scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CandidateFamilyMasks:
    """Boolean candidate masks for each structured action-family group."""

    play: Tensor
    hand: Tensor
    move: Tensor
    attack: Tensor
    slot: Tensor
    index: Tensor
    default: Tensor


def partition_candidate_family_masks(
    family_ids: Tensor,
    *,
    play_character_family_id: int,
    hand_family_ids: tuple[int, ...],
    main_move_family_id: int,
    attack_family_id: int,
    slot_family_ids: tuple[int, ...],
    index_family_ids: tuple[int, ...],
) -> CandidateFamilyMasks:
    """Partition candidate rows into structured scoring groups."""

    play_mask = family_ids == play_character_family_id if play_character_family_id >= 0 else torch.zeros_like(family_ids, dtype=torch.bool)
    hand_mask = torch.zeros_like(play_mask)
    for family_id in hand_family_ids:
        if family_id < 0:
            continue
        hand_mask |= family_ids == family_id
    move_mask = family_ids == main_move_family_id if main_move_family_id >= 0 else torch.zeros_like(play_mask)
    attack_mask = family_ids == attack_family_id if attack_family_id >= 0 else torch.zeros_like(play_mask)
    slot_mask = torch.zeros_like(play_mask)
    for family_id in slot_family_ids:
        if family_id < 0:
            continue
        slot_mask |= family_ids == family_id
    index_mask = torch.zeros_like(play_mask)
    for family_id in index_family_ids:
        if family_id < 0:
            continue
        index_mask |= family_ids == family_id
    default_mask = ~(play_mask | hand_mask | move_mask | attack_mask | slot_mask | index_mask)
    return CandidateFamilyMasks(
        play=play_mask,
        hand=hand_mask,
        move=move_mask,
        attack=attack_mask,
        slot=slot_mask,
        index=index_mask,
        default=default_mask,
    )


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
    masks = partition_candidate_family_masks(
        family_ids,
        play_character_family_id=play_character_family_id,
        hand_family_ids=hand_family_ids,
        main_move_family_id=main_move_family_id,
        attack_family_id=attack_family_id,
        slot_family_ids=slot_family_ids,
        index_family_ids=index_family_ids,
    )

    def _indices(mask: Tensor) -> Tensor:
        if not torch.any(mask):
            return torch.zeros((0,), device=device, dtype=torch.long)
        return torch.nonzero(mask, as_tuple=False).squeeze(1)

    return (
        _indices(masks.play),
        _indices(masks.hand),
        _indices(masks.move),
        _indices(masks.attack),
        _indices(masks.slot),
        _indices(masks.index),
        _indices(masks.default),
    )


__all__ = [
    "CandidateFamilyMasks",
    "partition_candidate_family_indices",
    "partition_candidate_family_masks",
]
