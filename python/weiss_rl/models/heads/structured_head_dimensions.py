"""Candidate feature dimensions for the structured legal-action head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredHeadDimensions:
    family_embed_dim: int
    slot_embed_dim: int
    card_embed_dim: int
    slot_context_dim: int
    state_width: int
    generic_embed_dim: int
    candidate_input_dim: int


@dataclass(frozen=True)
class StructuredCandidateFeatureOffsets:
    family: int
    hand_card: int
    stage_slot: int
    from_slot: int
    to_slot: int
    attack_slot: int
    attack_type: int
    play_target_context: int
    move_source_context: int
    move_target_context: int
    attack_source_context: int
    defender_context: int
    numeric: int


def install_candidate_feature_offsets(
    head: Any,
    *,
    dimensions: StructuredHeadDimensions,
    offsets: StructuredCandidateFeatureOffsets,
) -> None:
    head._slot_context_dim = dimensions.slot_context_dim
    head._family_feature_offset = offsets.family
    head._hand_card_feature_offset = offsets.hand_card
    head._stage_slot_feature_offset = offsets.stage_slot
    head._from_slot_feature_offset = offsets.from_slot
    head._to_slot_feature_offset = offsets.to_slot
    head._attack_slot_feature_offset = offsets.attack_slot
    head._attack_type_feature_offset = offsets.attack_type
    head._play_target_context_offset = offsets.play_target_context
    head._move_source_context_offset = offsets.move_source_context
    head._move_target_context_offset = offsets.move_target_context
    head._attack_source_context_offset = offsets.attack_source_context
    head._defender_context_offset = offsets.defender_context
    head._numeric_feature_offset = offsets.numeric
    head._candidate_input_dim = int(dimensions.candidate_input_dim)


def resolve_structured_head_dimensions(action_feature_width: int) -> StructuredHeadDimensions:
    family_embed_dim = max(12, min(48, action_feature_width // 3))
    slot_embed_dim = max(8, min(24, action_feature_width // 5))
    card_embed_dim = max(16, min(64, action_feature_width // 2))
    slot_context_dim = max(24, action_feature_width // 2)
    state_width = max(32, int(action_feature_width))
    generic_embed_dim = max(8, min(24, action_feature_width // 5))
    candidate_input_dim = family_embed_dim + card_embed_dim + slot_embed_dim * 5 + slot_context_dim * 5 + 11
    return StructuredHeadDimensions(
        family_embed_dim=family_embed_dim,
        slot_embed_dim=slot_embed_dim,
        card_embed_dim=card_embed_dim,
        slot_context_dim=slot_context_dim,
        state_width=state_width,
        generic_embed_dim=generic_embed_dim,
        candidate_input_dim=candidate_input_dim,
    )


def resolve_candidate_feature_offsets(dimensions: StructuredHeadDimensions) -> StructuredCandidateFeatureOffsets:
    family = 0
    hand_card = family + dimensions.family_embed_dim
    stage_slot = hand_card + dimensions.card_embed_dim
    from_slot = stage_slot + dimensions.slot_embed_dim
    to_slot = from_slot + dimensions.slot_embed_dim
    attack_slot = to_slot + dimensions.slot_embed_dim
    attack_type = attack_slot + dimensions.slot_embed_dim
    play_target_context = attack_type + dimensions.slot_embed_dim
    move_source_context = play_target_context + dimensions.slot_context_dim
    move_target_context = move_source_context + dimensions.slot_context_dim
    attack_source_context = move_target_context + dimensions.slot_context_dim
    defender_context = attack_source_context + dimensions.slot_context_dim
    numeric = defender_context + dimensions.slot_context_dim
    return StructuredCandidateFeatureOffsets(
        family=family,
        hand_card=hand_card,
        stage_slot=stage_slot,
        from_slot=from_slot,
        to_slot=to_slot,
        attack_slot=attack_slot,
        attack_type=attack_type,
        play_target_context=play_target_context,
        move_source_context=move_source_context,
        move_target_context=move_target_context,
        attack_source_context=attack_source_context,
        defender_context=defender_context,
        numeric=numeric,
    )


__all__ = [
    "StructuredCandidateFeatureOffsets",
    "StructuredHeadDimensions",
    "install_candidate_feature_offsets",
    "resolve_candidate_feature_offsets",
    "resolve_structured_head_dimensions",
]
