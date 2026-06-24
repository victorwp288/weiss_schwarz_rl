"""Structured auxiliary learner metadata and packed legal-action helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.public_heuristic_profiles import (
    SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES,
    SUPPORTED_PUBLIC_HEURISTIC_PROFILES,
    active_public_heuristic_profiles,
    mix_public_heuristic_profile_logits,
    normalize_public_heuristic_profile_mode,
    normalize_public_heuristic_profiles,
    score_public_heuristic_target_logits,
    score_public_teacher_target_logits,
    selected_public_heuristic_profiles,
)
from weiss_rl.learners.structured_legal_view import (
    PackedStructuredLegalView,
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
)


@dataclass(frozen=True, slots=True)
class StructuredCatalogMetadata:
    family_names: tuple[str, ...]
    attack_type_names: tuple[str, ...]
    family_ids: tuple[int, ...]
    hand_indices: tuple[int, ...]
    play_slots: tuple[int, ...]
    move_from_slots: tuple[int, ...]
    move_to_slots: tuple[int, ...]
    attack_slots: tuple[int, ...]
    attack_types: tuple[int, ...]
    main_move_02_action_id: int | None


@lru_cache(maxsize=8)
def structured_catalog_metadata(action_catalog: ActionCatalog) -> StructuredCatalogMetadata:
    """Build stable per-action structured metadata for auxiliary losses."""
    family_names = tuple(family.name for family in action_catalog.families)
    attack_type_names = tuple(action_catalog.attack_type_names)
    family_index = {name: index for index, name in enumerate(family_names)}
    action_space = int(action_catalog.action_space_size)
    family_ids = np.full((action_space,), -1, dtype=np.int64)
    hand_indices = np.full((action_space,), -1, dtype=np.int64)
    play_slots = np.full((action_space,), -1, dtype=np.int64)
    move_from_slots = np.full((action_space,), -1, dtype=np.int64)
    move_to_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_types = np.full((action_space,), -1, dtype=np.int64)
    main_move_02_action_id: int | None = None
    attack_type_index = {name: index for index, name in enumerate(attack_type_names)}
    for action_id in range(action_space):
        decoded = action_catalog.decode(action_id)
        family_ids[action_id] = int(family_index.get(decoded.family, -1))
        if decoded.hand_index is not None:
            hand_indices[action_id] = int(decoded.hand_index)
        if decoded.family == "main_play_character" and decoded.stage_slot is not None:
            play_slots[action_id] = int(decoded.stage_slot)
        if decoded.family == "main_move" and decoded.from_slot is not None:
            move_from_slots[action_id] = int(decoded.from_slot)
        if decoded.family == "main_move" and decoded.to_slot is not None:
            move_to_slots[action_id] = int(decoded.to_slot)
        if decoded.family == "attack":
            if decoded.slot is not None:
                attack_slots[action_id] = int(decoded.slot)
            if decoded.attack_type is not None:
                attack_types[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
        if decoded.family == "main_move" and decoded.from_slot == 0 and decoded.to_slot == 2:
            main_move_02_action_id = int(action_id)
    return StructuredCatalogMetadata(
        family_names=family_names,
        attack_type_names=attack_type_names,
        family_ids=tuple(int(value) for value in family_ids.tolist()),
        hand_indices=tuple(int(value) for value in hand_indices.tolist()),
        play_slots=tuple(int(value) for value in play_slots.tolist()),
        move_from_slots=tuple(int(value) for value in move_from_slots.tolist()),
        move_to_slots=tuple(int(value) for value in move_to_slots.tolist()),
        attack_slots=tuple(int(value) for value in attack_slots.tolist()),
        attack_types=tuple(int(value) for value in attack_types.tolist()),
        main_move_02_action_id=main_move_02_action_id,
    )


def structured_group_lookup(action_catalog: ActionCatalog, *, device: torch.device) -> dict[str, Any]:
    metadata = structured_catalog_metadata(action_catalog)
    family_names = metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    attack_type_names = metadata.attack_type_names

    return {
        "family_ids": torch.as_tensor(metadata.family_ids, dtype=torch.long, device=device),
        "play_slots": torch.as_tensor(metadata.play_slots, dtype=torch.long, device=device),
        "move_to_slots": torch.as_tensor(metadata.move_to_slots, dtype=torch.long, device=device),
        "attack_slots": torch.as_tensor(metadata.attack_slots, dtype=torch.long, device=device),
        "attack_types": torch.as_tensor(metadata.attack_types, dtype=torch.long, device=device),
        "family_names": family_names,
        "family_index": family_index,
        "attack_type_names": attack_type_names,
    }


def dense_group_log_probs(
    *,
    masked_logits: Tensor,
    group_ids: Tensor,
    group_count: int,
) -> Tensor:
    group_scores = torch.full(
        (masked_logits.shape[0], int(group_count)),
        -1.0e9,
        dtype=masked_logits.dtype,
        device=masked_logits.device,
    )
    for group_id in range(int(group_count)):
        group_mask = group_ids == int(group_id)
        if not bool(group_mask.any().item()):
            continue
        group_scores[:, group_id] = torch.logsumexp(
            torch.where(group_mask.unsqueeze(0), masked_logits, torch.full_like(masked_logits, -1.0e9)),
            dim=1,
        )
    row_log_z = torch.logsumexp(masked_logits, dim=1, keepdim=True)
    return group_scores - row_log_z


def resolve_public_heuristic_family_ids(
    *,
    family_names: tuple[str, ...],
    requested_families: tuple[str, ...],
) -> tuple[int, ...]:
    """Resolve configured public-heuristic family names to catalog ids."""
    normalized = tuple(str(name).strip() for name in requested_families if str(name).strip())
    if not normalized:
        return ()
    family_index = {name: index for index, name in enumerate(family_names)}
    missing = sorted({name for name in normalized if name not in family_index})
    if missing:
        raise ValueError("teacher_public_heuristic_families contains unknown action families: " + ", ".join(missing))
    return tuple(int(family_index[name]) for name in normalized)


__all__ = [
    "SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES",
    "SUPPORTED_PUBLIC_HEURISTIC_PROFILES",
    "PackedStructuredLegalView",
    "StructuredCatalogMetadata",
    "active_public_heuristic_profiles",
    "dense_group_log_probs",
    "mix_public_heuristic_profile_logits",
    "normalize_public_heuristic_profile_mode",
    "normalize_public_heuristic_profiles",
    "packed_group_log_probs",
    "packed_soft_target_cross_entropy",
    "packed_structured_legal_view",
    "resolve_public_heuristic_family_ids",
    "score_public_heuristic_target_logits",
    "score_public_teacher_target_logits",
    "selected_public_heuristic_profiles",
    "structured_catalog_metadata",
    "structured_group_lookup",
]
