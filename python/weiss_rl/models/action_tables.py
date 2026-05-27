"""Action lookup tables for structured policy heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class StructuredActionComponentTables:
    family_ids: np.ndarray
    action_arg0: np.ndarray
    action_arg1: np.ndarray
    hand_indices: np.ndarray
    stage_slots: np.ndarray
    from_slots: np.ndarray
    to_slots: np.ndarray
    attack_slots: np.ndarray
    attack_types: np.ndarray
    generic_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FactorizedActionLookupTables:
    family_arg_kind: np.ndarray
    family_arg0_size: np.ndarray
    family_arg1_size: np.ndarray
    family_noarg_action_ids: np.ndarray
    one_arg_action_ids: np.ndarray
    two_arg_action_ids: np.ndarray
    max_arg0: int
    max_arg1: int
    slot_family_ids: tuple[int, ...]
    index_family_ids: tuple[int, ...]


def build_structured_action_component_tables(
    *,
    action_catalog: ActionCatalog,
    action_dim: int,
    family_index: Mapping[str, int],
    attack_type_index: Mapping[str, int],
) -> StructuredActionComponentTables:
    family_ids = np.zeros((int(action_dim),), dtype=np.int64)
    action_arg0 = np.full((int(action_dim),), -1, dtype=np.int64)
    action_arg1 = np.full((int(action_dim),), -1, dtype=np.int64)
    hand_indices = np.full((int(action_dim),), -1, dtype=np.int64)
    stage_slots = np.full((int(action_dim),), -1, dtype=np.int64)
    from_slots = np.full((int(action_dim),), -1, dtype=np.int64)
    to_slots = np.full((int(action_dim),), -1, dtype=np.int64)
    attack_slots = np.full((int(action_dim),), -1, dtype=np.int64)
    attack_types = np.full((int(action_dim),), -1, dtype=np.int64)
    generic_indices = np.full((int(action_dim),), -1, dtype=np.int64)
    for action_id in range(int(action_dim)):
        decoded = action_catalog.decode(action_id)
        family_ids[action_id] = int(family_index.get(decoded.family, 0))
        if decoded.hand_index is not None:
            action_arg0[action_id] = int(decoded.hand_index)
            hand_indices[action_id] = int(decoded.hand_index)
        if decoded.stage_slot is not None:
            action_arg1[action_id] = int(decoded.stage_slot)
            stage_slots[action_id] = int(decoded.stage_slot)
        if decoded.from_slot is not None:
            action_arg0[action_id] = int(decoded.from_slot)
            from_slots[action_id] = int(decoded.from_slot)
        if decoded.to_slot is not None:
            action_arg1[action_id] = int(decoded.to_slot)
            to_slots[action_id] = int(decoded.to_slot)
        if decoded.slot is not None:
            action_arg0[action_id] = int(decoded.slot)
            attack_slots[action_id] = int(decoded.slot)
        if decoded.attack_type is not None:
            action_arg1[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
            attack_types[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
        if decoded.index is not None:
            action_arg0[action_id] = int(decoded.index)
            generic_indices[action_id] = int(decoded.index)
    return StructuredActionComponentTables(
        family_ids=family_ids,
        action_arg0=action_arg0,
        action_arg1=action_arg1,
        hand_indices=hand_indices,
        stage_slots=stage_slots,
        from_slots=from_slots,
        to_slots=to_slots,
        attack_slots=attack_slots,
        attack_types=attack_types,
        generic_indices=generic_indices,
    )


def build_factorized_action_lookup_tables(
    *,
    action_dim: int,
    family_count: int,
    family_index: Mapping[str, int],
    component_tables: StructuredActionComponentTables,
) -> FactorizedActionLookupTables:
    family_arg_kind = np.zeros((int(family_count),), dtype=np.int64)
    hand_family_names = {
        "mulligan_select",
        "clock_from_hand",
        "main_play_event",
        "climax_play",
    }
    slot_family_names = {"encore_pay", "encore_decline"}
    index_family_names = {"level_up", "trigger_order", "choice_select"}
    for family_name, family_id in family_index.items():
        if family_name in hand_family_names:
            family_arg_kind[family_id] = 1
        elif family_name == "main_play_character":
            family_arg_kind[family_id] = 2
        elif family_name == "main_move":
            family_arg_kind[family_id] = 3
        elif family_name == "attack":
            family_arg_kind[family_id] = 4
        elif family_name in slot_family_names:
            family_arg_kind[family_id] = 5
        elif family_name in index_family_names:
            family_arg_kind[family_id] = 6
    family_arg0_size = np.zeros((int(family_count),), dtype=np.int64)
    family_arg1_size = np.zeros((int(family_count),), dtype=np.int64)
    family_noarg_action_ids = np.full((int(family_count),), -1, dtype=np.int64)
    for action_id in range(int(action_dim)):
        family_id = int(component_tables.family_ids[action_id])
        arg0 = int(component_tables.action_arg0[action_id])
        arg1 = int(component_tables.action_arg1[action_id])
        if arg0 < 0 and arg1 < 0:
            family_noarg_action_ids[family_id] = action_id
            continue
        if arg0 >= 0:
            family_arg0_size[family_id] = max(family_arg0_size[family_id], arg0 + 1)
        if arg1 >= 0:
            family_arg1_size[family_id] = max(family_arg1_size[family_id], arg1 + 1)
    max_arg0 = max(int(family_arg0_size.max()) if family_arg0_size.size else 0, 1)
    max_arg1 = max(int(family_arg1_size.max()) if family_arg1_size.size else 0, 1)
    one_arg_action_ids = np.full((int(family_count), max_arg0), -1, dtype=np.int64)
    two_arg_action_ids = np.full((int(family_count), max_arg0, max_arg1), -1, dtype=np.int64)
    for action_id in range(int(action_dim)):
        family_id = int(component_tables.family_ids[action_id])
        arg0 = int(component_tables.action_arg0[action_id])
        arg1 = int(component_tables.action_arg1[action_id])
        if arg0 < 0 and arg1 < 0:
            continue
        if arg0 >= 0 and arg1 < 0:
            one_arg_action_ids[family_id, arg0] = action_id
        elif arg0 >= 0 and arg1 >= 0:
            two_arg_action_ids[family_id, arg0, arg1] = action_id
    return FactorizedActionLookupTables(
        family_arg_kind=family_arg_kind,
        family_arg0_size=family_arg0_size,
        family_arg1_size=family_arg1_size,
        family_noarg_action_ids=family_noarg_action_ids,
        one_arg_action_ids=one_arg_action_ids,
        two_arg_action_ids=two_arg_action_ids,
        max_arg0=max_arg0,
        max_arg1=max_arg1,
        slot_family_ids=tuple(int(family_index[name]) for name in sorted(slot_family_names) if name in family_index),
        index_family_ids=tuple(int(family_index[name]) for name in sorted(index_family_names) if name in family_index),
    )
