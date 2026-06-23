"""Public observation decoding for HeuristicPublicPolicy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return list(value)


def _coerce_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{context} must be an int-compatible value")
    return int(value)


@dataclass(frozen=True, slots=True)
class StageSlotPublic:
    occupied: bool
    has_attacked: bool
    power: int
    effective_soul: int
    side_attack_allowed: bool


@dataclass(frozen=True, slots=True)
class PublicBoardState:
    self_level_count: int
    self_clock_count: int
    self_stage: tuple[StageSlotPublic, ...]
    opponent_stage: tuple[StageSlotPublic, ...]
    choice_page_start: int
    choice_total: int


@dataclass(frozen=True, slots=True)
class PlayerPublicObservationLayout:
    level_count_index: int
    clock_count_index: int
    stage_base: int
    stage_slot_width: int
    stage_slot_count: int

    def parse_stage(self, obs_row: np.ndarray) -> tuple[StageSlotPublic, ...]:
        slots: list[StageSlotPublic] = []
        for slot_index in range(self.stage_slot_count):
            base = self.stage_base + slot_index * self.stage_slot_width
            card_id = int(obs_row[base])
            slots.append(
                StageSlotPublic(
                    occupied=card_id != 0,
                    has_attacked=bool(int(obs_row[base + 2])),
                    power=int(obs_row[base + 3]),
                    effective_soul=int(obs_row[base + 5]),
                    side_attack_allowed=bool(int(obs_row[base + 6])),
                )
            )
        return tuple(slots)


@dataclass(frozen=True, slots=True)
class PublicObservationLayout:
    obs_len: int
    choice_page_start_index: int
    choice_total_index: int
    self_player: PlayerPublicObservationLayout
    opponent_player: PlayerPublicObservationLayout

    @classmethod
    def from_spec_bundle(cls, spec_bundle: Mapping[str, object], *, stage_slot_count: int) -> PublicObservationLayout:
        observation_spec = _require_mapping(spec_bundle.get("observation"), context="spec_bundle.observation")
        if not bool(observation_spec.get("self_first", False)):
            raise ValueError("B2 HeuristicPublic requires a self-first observation layout")

        header_fields = _require_sequence(
            observation_spec.get("header_fields"),
            context="spec_bundle.observation.header_fields",
        )
        header_indices: dict[str, int] = {}
        for field in header_fields:
            field_mapping = _require_mapping(field, context="spec_bundle.observation.header_fields[]")
            header_indices[str(field_mapping["name"])] = _coerce_int(
                field_mapping["index"],
                context=f"spec_bundle.observation.header_fields[{field_mapping['name']!r}].index",
            )

        player_blocks = _require_sequence(
            observation_spec.get("player_blocks"),
            context="spec_bundle.observation.player_blocks",
        )
        if len(player_blocks) < 2:
            raise ValueError("B2 HeuristicPublic requires two player blocks in the observation spec")

        return cls(
            obs_len=_coerce_int(observation_spec["obs_len"], context="spec_bundle.observation.obs_len"),
            choice_page_start_index=int(header_indices["choice_page_start"]),
            choice_total_index=int(header_indices["choice_total"]),
            self_player=_parse_player_layout(player_blocks[0], stage_slot_count=stage_slot_count),
            opponent_player=_parse_player_layout(player_blocks[1], stage_slot_count=stage_slot_count),
        )

    def parse_public_board(self, obs_row: np.ndarray) -> PublicBoardState:
        flat_obs = np.asarray(obs_row, dtype=np.int32).reshape(-1)
        if flat_obs.shape[0] < self.obs_len:
            raise ValueError(f"observation row is too short ({flat_obs.shape[0]} < {self.obs_len})")
        return PublicBoardState(
            self_level_count=int(flat_obs[self.self_player.level_count_index]),
            self_clock_count=int(flat_obs[self.self_player.clock_count_index]),
            self_stage=self.self_player.parse_stage(flat_obs),
            opponent_stage=self.opponent_player.parse_stage(flat_obs),
            choice_page_start=int(flat_obs[self.choice_page_start_index]),
            choice_total=int(flat_obs[self.choice_total_index]),
        )


def _parse_player_layout(item: object, *, stage_slot_count: int) -> PlayerPublicObservationLayout:
    block = _require_mapping(item, context="spec_bundle.observation.player_blocks[]")
    base = _coerce_int(block["base"], context="spec_bundle.observation.player_blocks[].base")
    slices = _require_sequence(
        block.get("slices"),
        context="spec_bundle.observation.player_blocks[].slices",
    )
    slices_by_name = {}
    for slice_item in slices:
        slice_mapping = _require_mapping(
            slice_item,
            context="spec_bundle.observation.player_blocks[].slices[]",
        )
        slices_by_name[str(slice_mapping["name"])] = slice_mapping
    stage_slice = _require_mapping(
        slices_by_name["stage"],
        context="spec_bundle.observation.player_blocks[].slices.stage",
    )
    stage_len = _coerce_int(
        stage_slice["len"],
        context="spec_bundle.observation.player_blocks[].slices.stage.len",
    )
    if stage_len % stage_slot_count != 0:
        raise ValueError(f"stage slice length {stage_len} is not divisible by stage slot count {stage_slot_count}")
    return PlayerPublicObservationLayout(
        level_count_index=base
        + _coerce_int(
            slices_by_name["level_count"]["start"],
            context="spec_bundle.observation.player_blocks[].slices.level_count.start",
        ),
        clock_count_index=base
        + _coerce_int(
            slices_by_name["clock_count"]["start"],
            context="spec_bundle.observation.player_blocks[].slices.clock_count.start",
        ),
        stage_base=base
        + _coerce_int(
            stage_slice["start"],
            context="spec_bundle.observation.player_blocks[].slices.stage.start",
        ),
        stage_slot_width=stage_len // stage_slot_count,
        stage_slot_count=stage_slot_count,
    )


__all__ = [
    "PlayerPublicObservationLayout",
    "PublicBoardState",
    "PublicObservationLayout",
    "StageSlotPublic",
]
