"""Legal-action metadata construction for packed runtime candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.runtime.components.shared import DEFAULT_ACTION_META_WIDTH


def action_catalog_indices(action_catalog: ActionCatalog) -> tuple[dict[str, int], dict[str, int]]:
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    return family_index, attack_type_index


def legal_action_meta_from_ids(
    legal_ids: np.ndarray,
    *,
    action_catalog: ActionCatalog | None,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    action_meta_width: int,
) -> np.ndarray | None:
    if action_catalog is None:
        return None
    legal_ids_array = np.asarray(legal_ids, dtype=np.uint32)
    unused = np.iinfo(np.uint16).max
    width = max(int(action_meta_width), DEFAULT_ACTION_META_WIDTH)
    rows = np.full((int(legal_ids_array.shape[0]), width), unused, dtype=np.uint16)
    for row_index, action_id in enumerate(legal_ids_array.astype(np.int64, copy=False).tolist()):
        decoded = action_catalog.decode(int(action_id))
        rows[row_index, 0] = np.uint16(family_index[decoded.family])
        if decoded.hand_index is not None:
            rows[row_index, 1] = np.uint16(decoded.hand_index)
        if decoded.stage_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.stage_slot)
        if decoded.from_slot is not None:
            rows[row_index, 1] = np.uint16(decoded.from_slot)
        if decoded.to_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.to_slot)
        if decoded.slot is not None:
            rows[row_index, 1] = np.uint16(decoded.slot)
        if decoded.attack_type is not None:
            rows[row_index, 2] = np.uint16(attack_type_index[decoded.attack_type])
        if decoded.index is not None:
            rows[row_index, 1] = np.uint16(decoded.index)
    return rows


def ensure_legal_action_meta(
    legal_ids: np.ndarray,
    legal_action_meta: np.ndarray | None,
    *,
    build_meta: Any,
) -> np.ndarray | None:
    if legal_action_meta is not None:
        return np.asarray(legal_action_meta, dtype=np.uint16)
    return build_meta(legal_ids)
