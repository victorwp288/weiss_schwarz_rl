from __future__ import annotations

import pytest
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.observation_layout import parse_observation_layout
from weiss_rl.models.observations.observation_contract import (
    build_structured_observation_contract,
    header_field_index,
    slice_by_name,
)


def _catalog(max_stage: int = 2) -> ActionCatalog:
    return ActionCatalog(
        action_space_size=8,
        pass_action_id=0,
        max_hand=3,
        max_stage=max_stage,
        attack_slot_count=2,
        attack_type_names=("front",),
        families=(),
    )


def _observation_spec() -> dict[str, object]:
    return {
        "obs_len": 20,
        "self_first": True,
        "sentinel_hidden": -9,
        "sentinel_empty_card": -2,
        "header_fields": [
            {"name": "choice_page_start", "index": 18},
            {"name": "choice_total", "index": 19},
        ],
        "player_blocks": [
            {
                "name": "self",
                "base": 0,
                "len": 8,
                "slices": [
                    {"name": "stage", "start": 0, "len": 4},
                    {"name": "hand", "start": 4, "len": 2},
                    {"name": "clock_count", "start": 6, "len": 1},
                    {"name": "level_count", "start": 7, "len": 1},
                ],
            },
            {
                "name": "opponent",
                "base": 8,
                "len": 8,
                "slices": [
                    {"name": "stage", "start": 0, "len": 4},
                    {"name": "waiting_room_top", "start": 4, "len": 2},
                ],
            },
        ],
    }


def test_build_structured_observation_contract_collects_stage_and_card_indices() -> None:
    contract = build_structured_observation_contract(_observation_spec(), action_catalog=_catalog(max_stage=2))

    assert contract.layout.self_first is True
    assert contract.self_stage is not None
    assert contract.self_stage.start == 0
    assert contract.opponent_stage is not None
    assert contract.opponent_stage.start == 8
    assert contract.self_hand is not None
    assert contract.self_hand.indices == (4, 5)
    assert contract.choice_page_start_index == 18
    assert contract.choice_total_index == 19
    assert contract.stage_slot_count == 2
    assert contract.sentinel_hidden == -9
    assert contract.sentinel_empty_card == -2
    assert contract.card_scalar_indices == (0, 2, 4, 5, 8, 10, 12, 13)


def test_slice_and_header_lookup_helpers_return_none_for_missing_names() -> None:
    layout = parse_observation_layout(_observation_spec())
    self_block = layout.player_blocks[0]

    assert slice_by_name(self_block, "stage") is not None
    assert slice_by_name(self_block, "missing") is None
    assert header_field_index(layout, "choice_total") == 19
    assert header_field_index(layout, "missing") is None


def test_build_structured_observation_contract_rejects_non_self_first_layout() -> None:
    spec = _observation_spec()
    spec["self_first"] = False

    with pytest.raises(ValueError, match="self-first"):
        build_structured_observation_contract(spec, action_catalog=_catalog())


def test_build_structured_observation_contract_rejects_bad_stage_width() -> None:
    spec = _observation_spec()
    spec["player_blocks"] = [
        {
            "name": "self",
            "base": 0,
            "len": 3,
            "slices": [{"name": "stage", "start": 0, "len": 3}],
        },
        {
            "name": "opponent",
            "base": 3,
            "len": 3,
            "slices": [{"name": "stage", "start": 0, "len": 3}],
        },
    ]

    with pytest.raises(ValueError, match="not divisible by stage slot count"):
        build_structured_observation_contract(spec, action_catalog=_catalog(max_stage=2))
