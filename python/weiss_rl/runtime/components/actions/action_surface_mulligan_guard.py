"""Mulligan-specific packed action-surface guard."""

from __future__ import annotations

import numpy as np

from weiss_rl.runtime.components.actions.action_surface_packed import (
    PackedActionSurfaceFilterResult,
    changed_filter_result,
    unchanged_filter_result,
)


def filter_mulligan_select_after_select_from_ids(
    *,
    obs: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
    last_action_arg0_index: int,
    mulligan_select_family_id: int,
    mulligan_confirm_family_id: int,
) -> PackedActionSurfaceFilterResult:
    """Remove further mulligan-select actions after a select action has occurred.

    Simulator v1 exposes mulligan as an iterative select/confirm surface. The
    selected-card set is not part of the public observation, while the same
    select actions remain legal. This optional guard lets the RL stack test the
    one-select-then-confirm abstraction without editing the simulator package.
    """

    ids_array = np.asarray(legal_ids, dtype=np.uint32)
    offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
    meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
    obs_array = np.asarray(obs)
    if (
        ids_array.ndim != 1
        or offsets_array.ndim != 1
        or offsets_array.size < 1
        or int(last_action_arg0_index) < 0
        or int(mulligan_select_family_id) < 0
        or int(mulligan_confirm_family_id) < 0
        or meta_array is None
        or meta_array.ndim != 2
        or meta_array.shape[0] != ids_array.shape[0]
        or meta_array.shape[1] < 1
        or obs_array.ndim != 2
        or obs_array.shape[0] != offsets_array.size - 1
        or int(last_action_arg0_index) >= obs_array.shape[1]
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
    last_action_arg0 = obs_array[:, int(last_action_arg0_index)]
    select_family = int(mulligan_select_family_id)
    confirm_family = int(mulligan_confirm_family_id)
    family_ids = meta_array[:, 0].astype(np.int64, copy=False)

    for row_index in range(offsets_array.size - 1):
        start = int(offsets_array[row_index])
        stop = int(offsets_array[row_index + 1])
        row_ids = ids_array[start:stop]
        row_meta = meta_array[start:stop]
        row_families = family_ids[start:stop]
        keep = np.ones((int(row_ids.shape[0]),), dtype=np.bool_)
        selected_before = int(last_action_arg0[row_index]) >= 0
        has_confirm = bool(np.any(row_families == confirm_family))
        has_select = bool(np.any(row_families == select_family))
        if selected_before and has_confirm and has_select:
            keep = row_families != select_family
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


__all__ = ["filter_mulligan_select_after_select_from_ids"]
