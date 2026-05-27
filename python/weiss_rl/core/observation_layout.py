"""Observation-layout parsing for contract-aware encoders and tooling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservationField:
    name: str
    index: int


@dataclass(frozen=True, slots=True)
class ObservationSlice:
    name: str
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True, slots=True)
class ObservationPlayerBlock:
    name: str
    base: int
    length: int
    slices: tuple[ObservationSlice, ...]

    @property
    def stop(self) -> int:
        return self.base + self.length

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.base, self.stop))


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    obs_len: int
    self_first: bool
    header_fields: tuple[ObservationField, ...]
    player_blocks: tuple[ObservationPlayerBlock, ...]
    tail_slices: tuple[ObservationSlice, ...]


def parse_observation_layout(value: Mapping[str, Any]) -> ObservationLayout:
    spec = _require_mapping(value, context="observation")
    obs_len = _require_int(spec.get("obs_len"), field_name="observation.obs_len", minimum=1)
    header_fields = _parse_header_fields(spec.get("header_fields"), obs_len=obs_len)
    player_blocks = _parse_player_blocks(spec.get("player_blocks"), obs_len=obs_len)
    tail_slices = _parse_tail_slices(spec.get("tail_slices"), obs_len=obs_len)
    return ObservationLayout(
        obs_len=obs_len,
        self_first=bool(spec.get("self_first", False)),
        header_fields=header_fields,
        player_blocks=player_blocks,
        tail_slices=tail_slices,
    )


def parse_observation_layout_from_spec_bundle(spec_bundle: Mapping[str, Any]) -> ObservationLayout:
    bundle = _require_mapping(spec_bundle, context="spec_bundle")
    observation = _require_mapping(bundle.get("observation"), context="spec_bundle.observation")
    return parse_observation_layout(observation)


def _parse_header_fields(value: object, *, obs_len: int) -> tuple[ObservationField, ...]:
    if value is None:
        return ()
    fields = _require_list(value, context="observation.header_fields")
    parsed: list[ObservationField] = []
    seen_names: set[str] = set()
    seen_indices: set[int] = set()
    for item in fields:
        field = _require_mapping(item, context="observation.header_fields[]")
        name = _require_text(field.get("name"), field_name="observation.header_fields[].name")
        index = _require_int(field.get("index"), field_name=f"observation.header_fields[{name!r}].index", minimum=0)
        if index >= obs_len:
            raise ValueError(f"observation.header_fields[{name!r}].index must be < obs_len ({obs_len}), got {index}")
        if name in seen_names:
            raise ValueError(f"observation.header_fields contains duplicate field name: {name!r}")
        if index in seen_indices:
            raise ValueError(f"observation.header_fields contains duplicate index: {index}")
        seen_names.add(name)
        seen_indices.add(index)
        parsed.append(ObservationField(name=name, index=index))
    return tuple(sorted(parsed, key=lambda field: (field.index, field.name)))


def _parse_player_blocks(value: object, *, obs_len: int) -> tuple[ObservationPlayerBlock, ...]:
    if value is None:
        return ()
    items = _require_list(value, context="observation.player_blocks")
    parsed: list[ObservationPlayerBlock] = []
    for block_index, item in enumerate(items):
        block = _require_mapping(item, context="observation.player_blocks[]")
        base = _require_int(block.get("base"), field_name="observation.player_blocks[].base", minimum=0)
        raw_name = block.get("name")
        name = (
            f"player_{block_index}"
            if raw_name is None
            else _require_text(raw_name, field_name="observation.player_blocks[].name")
        )
        slices_raw = _require_list(block.get("slices", []), context="observation.player_blocks[].slices")
        slices: list[ObservationSlice] = []
        max_relative_stop = 0
        seen_slice_names: set[str] = set()
        for slice_item in slices_raw:
            slice_mapping = _require_mapping(slice_item, context="observation.player_blocks[].slices[]")
            slice_name = _require_text(
                slice_mapping.get("name"),
                field_name="observation.player_blocks[].slices[].name",
            )
            start = _require_int(
                slice_mapping.get("start"),
                field_name=f"observation.player_blocks[{name!r}].slices[{slice_name!r}].start",
                minimum=0,
            )
            length = _require_int(
                slice_mapping.get("len"),
                field_name=f"observation.player_blocks[{name!r}].slices[{slice_name!r}].len",
                minimum=1,
            )
            if slice_name in seen_slice_names:
                raise ValueError(f"observation.player_blocks[{name!r}] contains duplicate slice name: {slice_name!r}")
            seen_slice_names.add(slice_name)
            max_relative_stop = max(max_relative_stop, start + length)
            slices.append(ObservationSlice(name=slice_name, start=base + start, length=length))

        explicit_length = block.get("len")
        block_length = (
            _require_int(explicit_length, field_name=f"observation.player_blocks[{name!r}].len", minimum=1)
            if explicit_length is not None
            else max_relative_stop
        )
        if block_length < 1:
            raise ValueError(f"observation.player_blocks[{name!r}] must define len>=1 or at least one non-empty slice")
        if block_length < max_relative_stop:
            raise ValueError(
                f"observation.player_blocks[{name!r}].len must cover all slices: "
                f"len={block_length}, required>={max_relative_stop}"
            )
        if base + block_length > obs_len:
            raise ValueError(
                f"observation.player_blocks[{name!r}] exceeds obs_len ({obs_len}): base={base}, len={block_length}"
            )
        parsed.append(
            ObservationPlayerBlock(
                name=name,
                base=base,
                length=block_length,
                slices=tuple(sorted(slices, key=lambda current: (current.start, current.name))),
            )
        )
    return tuple(parsed)


def _parse_tail_slices(value: object, *, obs_len: int) -> tuple[ObservationSlice, ...]:
    if value is None:
        return ()
    items = _require_list(value, context="observation.tail_slices")
    parsed: list[ObservationSlice] = []
    seen_names: set[str] = set()
    for item in items:
        slice_mapping = _require_mapping(item, context="observation.tail_slices[]")
        name = _require_text(slice_mapping.get("name"), field_name="observation.tail_slices[].name")
        start = _require_int(
            slice_mapping.get("start"),
            field_name=f"observation.tail_slices[{name!r}].start",
            minimum=0,
        )
        length = _require_int(
            slice_mapping.get("len"),
            field_name=f"observation.tail_slices[{name!r}].len",
            minimum=1,
        )
        if start + length > obs_len:
            raise ValueError(
                f"observation.tail_slices[{name!r}] exceeds obs_len ({obs_len}): start={start}, len={length}"
            )
        if name in seen_names:
            raise ValueError(f"observation.tail_slices contains duplicate slice name: {name!r}")
        seen_names.add(name)
        parsed.append(ObservationSlice(name=name, start=start, length=length))
    return tuple(sorted(parsed, key=lambda current: (current.start, current.name)))


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return list(value)


def _require_int(value: object, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
