from __future__ import annotations

import numpy as np
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.models.action_tables import (
    build_factorized_action_lookup_tables,
    build_structured_action_component_tables,
)


def _catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 46,
                "pass_action_id": 40,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 9},
                    {"name": "main_move", "base": 19, "count": 20},
                    {"name": "climax_play", "base": 39, "count": 1},
                    {"name": "pass", "base": 40, "count": 1},
                    {"name": "choice_select", "base": 41, "count": 3},
                    {"name": "choice_prev_page", "base": 44, "count": 1},
                    {"name": "choice_next_page", "base": 45, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_build_structured_action_component_tables_decodes_action_catalog() -> None:
    catalog = _catalog()
    family_index = {family.name: index for index, family in enumerate(catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(catalog.attack_type_names)}

    tables = build_structured_action_component_tables(
        action_catalog=catalog,
        action_dim=catalog.action_space_size,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )

    assert tables.family_ids[[7, 14, 25, 39, 40, 43]].tolist() == [0, 1, 2, 3, 4, 5]
    assert tables.action_arg0[[7, 14, 25, 39, 40, 43]].tolist() == [1, 1, 1, 0, -1, 2]
    assert tables.action_arg1[[7, 14, 25, 39, 40, 43]].tolist() == [2, 1, 3, -1, -1, -1]
    assert tables.hand_indices[[7, 39]].tolist() == [1, 0]
    assert tables.stage_slots[7] == 2
    assert tables.from_slots[25] == 1
    assert tables.to_slots[25] == 3
    assert tables.attack_slots[14] == 1
    assert tables.attack_types[14] == 1
    assert tables.generic_indices[43] == 2


def test_build_factorized_action_lookup_tables_preserves_family_argument_contracts() -> None:
    catalog = _catalog()
    family_index = {family.name: index for index, family in enumerate(catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(catalog.attack_type_names)}
    component_tables = build_structured_action_component_tables(
        action_catalog=catalog,
        action_dim=catalog.action_space_size,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )

    tables = build_factorized_action_lookup_tables(
        action_dim=catalog.action_space_size,
        family_count=len(catalog.families),
        family_index=family_index,
        component_tables=component_tables,
    )

    assert tables.family_arg_kind.tolist() == [2, 4, 3, 1, 0, 6, 0, 0]
    assert tables.family_noarg_action_ids[family_index["pass"]] == 40
    assert tables.family_noarg_action_ids[family_index["choice_prev_page"]] == 44
    assert tables.family_noarg_action_ids[family_index["choice_next_page"]] == 45
    assert tables.one_arg_action_ids[family_index["climax_play"], 0] == 39
    assert tables.one_arg_action_ids[family_index["choice_select"], 2] == 43
    assert tables.two_arg_action_ids[family_index["main_play_character"], 1, 2] == 7
    assert tables.two_arg_action_ids[family_index["main_move"], 1, 3] == 25
    assert tables.two_arg_action_ids[family_index["attack"], 1, 1] == 14
    assert tables.index_family_ids == (family_index["choice_select"],)
    assert tables.slot_family_ids == ()
    assert tables.max_arg0 >= 3
    assert tables.max_arg1 >= 4
    assert np.all(tables.one_arg_action_ids[family_index["pass"]] == -1)


def test_factorized_lookup_argument_kinds_match_action_catalog_decode_contract() -> None:
    catalog = _catalog()
    family_index = {family.name: index for index, family in enumerate(catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(catalog.attack_type_names)}
    component_tables = build_structured_action_component_tables(
        action_catalog=catalog,
        action_dim=catalog.action_space_size,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )

    tables = build_factorized_action_lookup_tables(
        action_dim=catalog.action_space_size,
        family_count=len(catalog.families),
        family_index=family_index,
        component_tables=component_tables,
    )

    no_arg_families = {
        decoded.family
        for action_id in range(catalog.action_space_size)
        if (decoded := catalog.decode(action_id)).hand_index is None
        and decoded.stage_slot is None
        and decoded.from_slot is None
        and decoded.to_slot is None
        and decoded.slot is None
        and decoded.attack_type is None
        and decoded.index is None
    }
    assert {"choice_prev_page", "choice_next_page"} <= no_arg_families
    for family_name in no_arg_families:
        family_id = family_index[family_name]
        assert tables.family_arg_kind[family_id] == 0
        assert tables.family_noarg_action_ids[family_id] >= 0
