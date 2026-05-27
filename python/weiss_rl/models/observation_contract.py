"""Structured observation contract helpers for model encoders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.observation_layout import (
    ObservationLayout,
    ObservationPlayerBlock,
    ObservationSlice,
    parse_observation_layout,
)

CARD_ID_VECTOR_SLICE_NAMES = frozenset(
    {
        "climax_top",
        "clock_top",
        "deck",
        "hand",
        "level_top",
        "resolution_top",
        "stock_top",
        "waiting_room_top",
    }
)


@dataclass(frozen=True, slots=True)
class StructuredObservationContract:
    layout: ObservationLayout
    self_stage: ObservationSlice | None
    opponent_stage: ObservationSlice | None
    self_hand: ObservationSlice | None
    self_level_count: ObservationSlice | None
    self_clock_count: ObservationSlice | None
    choice_page_start_index: int | None
    choice_total_index: int | None
    stage_slot_count: int
    sentinel_hidden: int
    sentinel_empty_card: int
    card_scalar_indices: tuple[int, ...]


def slice_by_name(block: ObservationPlayerBlock, name: str) -> ObservationSlice | None:
    for current in block.slices:
        if current.name == name:
            return current
    return None


def header_field_index(layout: ObservationLayout, name: str) -> int | None:
    for field in layout.header_fields:
        if field.name == name:
            return int(field.index)
    return None


def build_structured_observation_contract(
    observation_spec: Mapping[str, Any],
    *,
    action_catalog: ActionCatalog,
) -> StructuredObservationContract:
    layout = parse_observation_layout(observation_spec)
    if not layout.self_first:
        raise ValueError("structured_v2 requires a self-first observation layout")
    if len(layout.player_blocks) < 2:
        raise ValueError("structured_v2 requires at least two player blocks in the observation layout")
    stage_slot_count = max(int(action_catalog.max_stage), 1)
    self_block = layout.player_blocks[0]
    opponent_block = layout.player_blocks[1]
    self_stage = slice_by_name(self_block, "stage")
    opponent_stage = slice_by_name(opponent_block, "stage")
    self_hand = slice_by_name(self_block, "hand")
    self_level_count = slice_by_name(self_block, "level_count")
    self_clock_count = slice_by_name(self_block, "clock_count")

    for stage_slice, stage_name in ((self_stage, "self"), (opponent_stage, "opponent")):
        if stage_slice is None:
            continue
        if stage_slice.length % stage_slot_count != 0:
            raise ValueError(
                f"structured_v2 {stage_name} stage slice length {stage_slice.length} "
                f"is not divisible by stage slot count {stage_slot_count}"
            )

    card_scalar_indices: set[int] = set()
    for block in layout.player_blocks:
        stage_slice = slice_by_name(block, "stage")
        if stage_slice is not None:
            slot_width = max(stage_slice.length // stage_slot_count, 1)
            for slot_index in range(stage_slot_count):
                card_scalar_indices.add(stage_slice.start + slot_index * slot_width)
        for current in block.slices:
            if current.name in CARD_ID_VECTOR_SLICE_NAMES:
                card_scalar_indices.update(current.indices)

    return StructuredObservationContract(
        layout=layout,
        self_stage=self_stage,
        opponent_stage=opponent_stage,
        self_hand=self_hand,
        self_level_count=self_level_count,
        self_clock_count=self_clock_count,
        choice_page_start_index=header_field_index(layout, "choice_page_start"),
        choice_total_index=header_field_index(layout, "choice_total"),
        stage_slot_count=stage_slot_count,
        sentinel_hidden=int(observation_spec.get("sentinel_hidden", -1)),
        sentinel_empty_card=int(observation_spec.get("sentinel_empty_card", 0)),
        card_scalar_indices=tuple(sorted(card_scalar_indices)),
    )
