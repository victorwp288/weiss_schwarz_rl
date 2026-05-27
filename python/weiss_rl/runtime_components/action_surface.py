"""RL-side action-surface guards for simulator decision quirks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PackedActionSurfaceFilterResult:
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    filtered_rows: int
    filtered_actions: int


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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
        )
    return PackedActionSurfaceFilterResult(
        legal_ids=(
            np.concatenate(filtered_ids, axis=0).astype(ids_array.dtype, copy=False)
            if filtered_ids
            else np.zeros((0,), dtype=ids_array.dtype)
        ),
        legal_offsets=filtered_offsets.astype(offsets_array.dtype, copy=False),
        legal_action_meta=(
            np.concatenate(filtered_meta, axis=0).astype(meta_array.dtype, copy=False)
            if filtered_meta
            else np.zeros((0, meta_array.shape[1]), dtype=meta_array.dtype)
        ),
        filtered_rows=filtered_rows,
        filtered_actions=filtered_actions,
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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
        )
    return PackedActionSurfaceFilterResult(
        legal_ids=(
            np.concatenate(filtered_ids, axis=0).astype(ids_array.dtype, copy=False)
            if filtered_ids
            else np.zeros((0,), dtype=ids_array.dtype)
        ),
        legal_offsets=filtered_offsets.astype(offsets_array.dtype, copy=False),
        legal_action_meta=(
            np.concatenate(filtered_meta, axis=0).astype(meta_array.dtype, copy=False)
            if filtered_meta
            else np.zeros((0, meta_array.shape[1]), dtype=meta_array.dtype)
        ),
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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
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
        return PackedActionSurfaceFilterResult(
            legal_ids=ids_array,
            legal_offsets=offsets_array,
            legal_action_meta=meta_array,
            filtered_rows=0,
            filtered_actions=0,
        )
    return PackedActionSurfaceFilterResult(
        legal_ids=(
            np.concatenate(filtered_ids, axis=0).astype(ids_array.dtype, copy=False)
            if filtered_ids
            else np.zeros((0,), dtype=ids_array.dtype)
        ),
        legal_offsets=filtered_offsets.astype(offsets_array.dtype, copy=False),
        legal_action_meta=(
            np.concatenate(filtered_meta, axis=0).astype(meta_array.dtype, copy=False)
            if filtered_meta
            else np.zeros((0, meta_array.shape[1]), dtype=meta_array.dtype)
        ),
        filtered_rows=filtered_rows,
        filtered_actions=filtered_actions,
    )


def filter_batch_mulligan_select_after_select(
    batch: Any,
    *,
    last_action_arg0_index: int,
    mulligan_select_family_id: int,
    mulligan_confirm_family_id: int,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        empty = PackedActionSurfaceFilterResult(
            legal_ids=np.zeros((0,), dtype=np.uint32),
            legal_offsets=np.zeros((1,), dtype=np.uint32),
            legal_action_meta=None,
            filtered_rows=0,
            filtered_actions=0,
        )
        return batch, empty
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_mulligan_select_after_select_from_ids(
        obs=np.asarray(batch.obs),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        last_action_arg0_index=int(last_action_arg0_index),
        mulligan_select_family_id=int(mulligan_select_family_id),
        mulligan_confirm_family_id=int(mulligan_confirm_family_id),
    )
    if result.filtered_actions <= 0:
        return batch, result
    return (
        replace(
            batch,
            ids_offsets=(result.legal_ids, result.legal_offsets),
            legal_action_meta=result.legal_action_meta,
        ),
        result,
    )


def filter_batch_main_move_only_rows_to_pass(
    batch: Any,
    *,
    pass_action_id: int,
    main_move_family_id: int,
    allow_main_move_only_rows: np.ndarray | None = None,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        empty = PackedActionSurfaceFilterResult(
            legal_ids=np.zeros((0,), dtype=np.uint32),
            legal_offsets=np.zeros((1,), dtype=np.uint32),
            legal_action_meta=None,
            filtered_rows=0,
            filtered_actions=0,
        )
        return batch, empty
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_main_move_only_rows_to_pass_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        pass_action_id=int(pass_action_id),
        main_move_family_id=int(main_move_family_id),
        allow_main_move_only_rows=allow_main_move_only_rows,
    )
    if result.filtered_actions <= 0:
        return batch, result
    return (
        replace(
            batch,
            ids_offsets=(result.legal_ids, result.legal_offsets),
            legal_action_meta=result.legal_action_meta,
        ),
        result,
    )


def filter_batch_pass_when_attack_available(
    batch: Any,
    *,
    pass_action_id: int,
    attack_family_id: int,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        empty = PackedActionSurfaceFilterResult(
            legal_ids=np.zeros((0,), dtype=np.uint32),
            legal_offsets=np.zeros((1,), dtype=np.uint32),
            legal_action_meta=None,
            filtered_rows=0,
            filtered_actions=0,
        )
        return batch, empty
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_pass_when_attack_available_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        pass_action_id=int(pass_action_id),
        attack_family_id=int(attack_family_id),
    )
    if result.filtered_actions <= 0:
        return batch, result
    return (
        replace(
            batch,
            ids_offsets=(result.legal_ids, result.legal_offsets),
            legal_action_meta=result.legal_action_meta,
        ),
        result,
    )


__all__ = [
    "PackedActionSurfaceFilterResult",
    "filter_batch_main_move_only_rows_to_pass",
    "filter_batch_mulligan_select_after_select",
    "filter_batch_pass_when_attack_available",
    "filter_main_move_only_rows_to_pass_from_ids",
    "filter_mulligan_select_after_select_from_ids",
    "filter_pass_when_attack_available_from_ids",
]
