"""Candidate action component resolution for structured legal scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CandidateComponentFamilyIds:
    play_character: int
    main_event: int
    clock_from_hand: int
    climax_play: int
    mulligan_select: int
    main_move: int
    attack: int
    choice_select: int
    level_up: int
    trigger_order: int


def resolve_candidate_components(
    candidate_ids: Tensor,
    candidate_meta: Tensor | None,
    *,
    family_ids_by_action: Tensor,
    hand_indices_by_action: Tensor,
    stage_slots_by_action: Tensor,
    from_slots_by_action: Tensor,
    to_slots_by_action: Tensor,
    attack_slots_by_action: Tensor,
    attack_types_by_action: Tensor,
    generic_indices_by_action: Tensor,
    meta_unused: int,
    family_ids: CandidateComponentFamilyIds,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if candidate_meta is None:
        return (
            family_ids_by_action.index_select(0, candidate_ids),
            hand_indices_by_action.index_select(0, candidate_ids),
            stage_slots_by_action.index_select(0, candidate_ids),
            from_slots_by_action.index_select(0, candidate_ids),
            to_slots_by_action.index_select(0, candidate_ids),
            attack_slots_by_action.index_select(0, candidate_ids),
            attack_types_by_action.index_select(0, candidate_ids),
            generic_indices_by_action.index_select(0, candidate_ids),
        )

    resolved_family_ids = candidate_meta[:, 0].to(dtype=torch.long)
    arg0 = candidate_meta[:, 1].to(dtype=torch.long)
    arg1 = candidate_meta[:, 2].to(dtype=torch.long)
    meta_unused_tensor = torch.full_like(arg0, int(meta_unused))
    arg0 = torch.where(arg0 == meta_unused_tensor, torch.full_like(arg0, -1), arg0)
    arg1 = torch.where(arg1 == meta_unused_tensor, torch.full_like(arg1, -1), arg1)

    hand_indices = torch.full_like(arg0, -1)
    hand_family_ids = (
        family_ids.play_character,
        family_ids.main_event,
        family_ids.clock_from_hand,
        family_ids.climax_play,
        family_ids.mulligan_select,
    )
    for family_id in hand_family_ids:
        if family_id < 0:
            continue
        family_mask = resolved_family_ids == family_id
        hand_indices[family_mask] = arg0[family_mask]

    stage_slots = torch.full_like(arg0, -1)
    if family_ids.play_character >= 0:
        play_mask = resolved_family_ids == family_ids.play_character
        stage_slots[play_mask] = arg1[play_mask]

    from_slots = torch.full_like(arg0, -1)
    to_slots = torch.full_like(arg0, -1)
    if family_ids.main_move >= 0:
        move_mask = resolved_family_ids == family_ids.main_move
        from_slots[move_mask] = arg0[move_mask]
        to_slots[move_mask] = arg1[move_mask]

    attack_slots = torch.full_like(arg0, -1)
    attack_types = torch.full_like(arg0, -1)
    if family_ids.attack >= 0:
        attack_mask = resolved_family_ids == family_ids.attack
        attack_slots[attack_mask] = arg0[attack_mask]
        attack_types[attack_mask] = arg1[attack_mask]

    generic_indices = torch.full_like(arg0, -1)
    generic_family_ids = (
        family_ids.choice_select,
        family_ids.level_up,
        family_ids.trigger_order,
    )
    for family_id in generic_family_ids:
        if family_id < 0:
            continue
        generic_mask = resolved_family_ids == family_id
        generic_indices[generic_mask] = arg0[generic_mask]

    return (
        resolved_family_ids,
        hand_indices,
        stage_slots,
        from_slots,
        to_slots,
        attack_slots,
        attack_types,
        generic_indices,
    )
