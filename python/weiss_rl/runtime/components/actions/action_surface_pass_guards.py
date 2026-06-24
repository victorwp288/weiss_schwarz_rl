"""Pass/main-move packed action-surface guards."""

from __future__ import annotations

import numpy as np

from weiss_rl.runtime.components.actions.action_surface_packed import (
    PackedActionSurfaceFilterResult,
    changed_filter_result,
    unchanged_filter_result,
)


def filter_main_move_only_rows_to_pass_from_ids(
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
    pass_action_id: int,
    main_move_family_id: int,
    allow_main_move_only_rows: np.ndarray | None = None,
) -> PackedActionSurfaceFilterResult:
    """Remove main-move choices when pass is the only non-movement option."""

    ids_array = np.asarray(legal_ids, dtype=np.uint32)
    offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
    meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
    allow_rows = None if allow_main_move_only_rows is None else np.asarray(allow_main_move_only_rows, dtype=np.bool_)
    if (
        ids_array.ndim != 1
        or offsets_array.ndim != 1
        or offsets_array.size < 1
        or int(pass_action_id) < 0
        or int(main_move_family_id) < 0
        or meta_array is None
        or meta_array.ndim != 2
        or meta_array.shape[0] != ids_array.shape[0]
        or meta_array.shape[1] < 1
        or (allow_rows is not None and allow_rows.shape != (offsets_array.size - 1,))
    ):
        return unchanged_filter_result(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
        )

    filtered_ids: list[np.ndarray] = []
    filtered_meta: list[np.ndarray] = []
    filtered_offsets = np.zeros_like(offsets_array)
    cursor = 0
    filtered_rows = 0
    filtered_actions = 0
    pass_id = int(pass_action_id)
    main_move_id = int(main_move_family_id)
    family_ids = meta_array[:, 0].astype(np.int64, copy=False)

    for row_index in range(offsets_array.size - 1):
        start = int(offsets_array[row_index])
        stop = int(offsets_array[row_index + 1])
        row_ids = ids_array[start:stop]
        row_meta = meta_array[start:stop]
        row_families = family_ids[start:stop]
        keep = np.ones((int(row_ids.shape[0]),), dtype=np.bool_)
        has_pass = bool(np.any(row_ids == pass_id))
        nonpass = row_ids != pass_id
        has_main_move = bool(np.any(nonpass & (row_families == main_move_id)))
        has_nonmove_nonpass = bool(np.any(nonpass & (row_families != main_move_id)))
        allow_row = bool(allow_rows[row_index]) if allow_rows is not None else False
        if has_pass and has_main_move and not has_nonmove_nonpass and not allow_row:
            keep = row_ids == pass_id
            removed = int(np.count_nonzero(~keep))
            if removed > 0 and bool(np.any(keep)):
                filtered_rows += 1
                filtered_actions += removed
            else:
                keep = np.ones_like(keep)
        kept_ids = row_ids[keep]
        kept_meta = row_meta[keep]
        filtered_ids.append(kept_ids)
        filtered_meta.append(kept_meta)
        cursor += int(kept_ids.shape[0])
        filtered_offsets[row_index + 1] = cursor

    if filtered_actions == 0:
        return unchanged_filter_result(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
        )
    return changed_filter_result(
        legal_ids=ids_array,
        legal_offsets=offsets_array,
        legal_action_meta=meta_array,
        filtered_ids=filtered_ids,
        filtered_meta=filtered_meta,
        filtered_offsets=filtered_offsets,
        filtered_rows=filtered_rows,
        filtered_actions=filtered_actions,
    )


def filter_pass_when_attack_available_from_ids(
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
    pass_action_id: int,
    attack_family_id: int,
) -> PackedActionSurfaceFilterResult:
    """Remove pass when attacking is available on the same decision row."""

    ids_array = np.asarray(legal_ids, dtype=np.uint32)
    offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
    meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
    if (
        ids_array.ndim != 1
        or offsets_array.ndim != 1
        or offsets_array.size < 1
        or int(pass_action_id) < 0
        or int(attack_family_id) < 0
        or meta_array is None
        or meta_array.ndim != 2
        or meta_array.shape[0] != ids_array.shape[0]
        or meta_array.shape[1] < 1
    ):
        return unchanged_filter_result(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
        )

    filtered_ids: list[np.ndarray] = []
    filtered_meta: list[np.ndarray] = []
    filtered_offsets = np.zeros_like(offsets_array)
    cursor = 0
    filtered_rows = 0
    filtered_actions = 0
    pass_id = int(pass_action_id)
    attack_id = int(attack_family_id)
    family_ids = meta_array[:, 0].astype(np.int64, copy=False)

    for row_index in range(offsets_array.size - 1):
        start = int(offsets_array[row_index])
        stop = int(offsets_array[row_index + 1])
        row_ids = ids_array[start:stop]
        row_meta = meta_array[start:stop]
        row_families = family_ids[start:stop]
        keep = np.ones((int(row_ids.shape[0]),), dtype=np.bool_)
        has_pass = bool(np.any(row_ids == pass_id))
        has_attack = bool(np.any(row_families == attack_id))
        if has_pass and has_attack:
            keep = row_ids != pass_id
            removed = int(np.count_nonzero(~keep))
            if removed > 0 and bool(np.any(keep)):
                filtered_rows += 1
                filtered_actions += removed
            else:
                keep = np.ones_like(keep)
        kept_ids = row_ids[keep]
        kept_meta = row_meta[keep]
        filtered_ids.append(kept_ids)
        filtered_meta.append(kept_meta)
        cursor += int(kept_ids.shape[0])
        filtered_offsets[row_index + 1] = cursor

    if filtered_actions == 0:
        return unchanged_filter_result(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
        )
    return changed_filter_result(
        legal_ids=ids_array,
        legal_offsets=offsets_array,
        legal_action_meta=meta_array,
        filtered_ids=filtered_ids,
        filtered_meta=filtered_meta,
        filtered_offsets=filtered_offsets,
        filtered_rows=filtered_rows,
        filtered_actions=filtered_actions,
    )


__all__ = [
    "filter_main_move_only_rows_to_pass_from_ids",
    "filter_pass_when_attack_available_from_ids",
]
