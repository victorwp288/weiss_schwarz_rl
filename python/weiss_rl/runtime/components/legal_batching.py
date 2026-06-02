from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components import shared as runtime_shared

_DEFAULT_ACTION_META_WIDTH = runtime_shared.DEFAULT_ACTION_META_WIDTH


def infer_packed_meta_width(unrolls: Sequence[Any]) -> int:
    for unroll in unrolls:
        if unroll.legal_actions.meta is not None:
            return int(np.asarray(unroll.legal_actions.meta).shape[1])
    return _DEFAULT_ACTION_META_WIDTH


def _unroll_legal_actions_are_packed(unroll: Any) -> bool:
    legal_actions = unroll.legal_actions
    return legal_actions.ids is not None and legal_actions.offsets is not None


def _unroll_legal_actions_as_mask(unroll: Any, *, action_space: int) -> np.ndarray:
    return unroll.legal_actions.to_mask(
        expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
        action_space=int(action_space),
    )


def _concatenate_legacy_packed_legal_actions(unrolls: Sequence[Any], *, action_space: int) -> LegalActionBatch:
    packed_ids: list[np.ndarray] = []
    packed_meta: list[np.ndarray] = []
    packed_offsets = [np.array([0], dtype=np.uint32)]
    any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
    for unroll in unrolls:
        legal_actions = unroll.legal_actions
        assert legal_actions.ids is not None and legal_actions.offsets is not None
        row_limit = int(unroll.obs.shape[0] * unroll.obs.shape[1])
        offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
        ids_limit = int(offsets[min(row_limit, max(offsets.size - 1, 0))])
        ids = np.asarray(legal_actions.ids[:ids_limit], dtype=np.uint32)
        offset_base = int(packed_offsets[-1][-1])
        packed_ids.append(ids)
        packed_offsets.append(np.asarray(offsets[1 : row_limit + 1] + offset_base, dtype=np.uint32))
        if any_meta and legal_actions.meta is not None:
            packed_meta.append(np.asarray(legal_actions.meta[:ids_limit], dtype=np.uint16))
    return LegalActionBatch.from_packed(
        np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
        np.concatenate(packed_offsets, axis=0),
        meta=(
            np.concatenate(packed_meta, axis=0)
            if packed_meta
            else (np.zeros((0, infer_packed_meta_width(unrolls)), dtype=np.uint16) if any_meta else None)
        ),
        action_space=int(action_space),
    )


def _concatenate_ordered_packed_legal_actions(unrolls: Sequence[Any], *, action_space: int) -> LegalActionBatch:
    total_time_steps = int(unrolls[0].obs.shape[0])
    for unroll in unrolls[1:]:
        if int(unroll.obs.shape[0]) != total_time_steps:
            raise RuntimeError("packed legal-action concatenation requires aligned unroll lengths")
    if not all(unroll.legal_actions.row_count == int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls):
        return _concatenate_legacy_packed_legal_actions(unrolls, action_space=action_space)

    total_ids = sum(int(np.asarray(unroll.legal_actions.ids, dtype=np.uint32).size) for unroll in unrolls)
    total_rows = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls)
    total_batch = sum(int(unroll.obs.shape[1]) for unroll in unrolls)
    ordered_packed_ids = np.empty((total_ids,), dtype=np.uint32)
    any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
    ordered_packed_meta = (
        np.empty((total_ids, infer_packed_meta_width(unrolls)), dtype=np.uint16) if any_meta and total_ids > 0 else None
    )
    ordered_packed_offsets = np.empty((total_rows + 1,), dtype=np.uint32)
    ordered_packed_offsets[0] = 0
    ordered_widths = np.empty((total_time_steps, total_batch), dtype=np.uint32)
    batch_offset = 0
    for unroll in unrolls:
        legal_actions = unroll.legal_actions
        assert legal_actions.offsets is not None
        env_count = int(unroll.obs.shape[1])
        widths = np.diff(np.asarray(legal_actions.offsets, dtype=np.uint32)).reshape(total_time_steps, env_count)
        ordered_widths[:, batch_offset : batch_offset + env_count] = widths
        batch_offset += env_count
    ordered_packed_offsets[1:] = np.cumsum(ordered_widths.reshape(-1), dtype=np.uint64).astype(np.uint32, copy=False)
    ids_offset = 0
    for time_index in range(total_time_steps):
        for unroll in unrolls:
            legal_actions = unroll.legal_actions
            assert legal_actions.ids is not None and legal_actions.offsets is not None
            env_count = int(unroll.obs.shape[1])
            row_base = int(time_index * env_count)
            offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
            ids = np.asarray(legal_actions.ids, dtype=np.uint32)
            meta = None if legal_actions.meta is None else np.asarray(legal_actions.meta, dtype=np.uint16)
            start = int(offsets[row_base])
            end = int(offsets[row_base + env_count])
            width = end - start
            if width > 0:
                ordered_packed_ids[ids_offset : ids_offset + width] = ids[start:end]
                if ordered_packed_meta is not None:
                    if meta is None:
                        ordered_packed_meta[ids_offset : ids_offset + width] = np.iinfo(np.uint16).max
                    else:
                        ordered_packed_meta[ids_offset : ids_offset + width] = meta[start:end]
            ids_offset += width

    return LegalActionBatch.from_packed(
        ordered_packed_ids[:ids_offset],
        ordered_packed_offsets,
        meta=None if ordered_packed_meta is None else ordered_packed_meta[:ids_offset],
        action_space=int(action_space),
    )


def _concatenate_mask_legal_actions(unrolls: Sequence[Any], *, action_space: int) -> LegalActionBatch:
    mask_parts = [_unroll_legal_actions_as_mask(unroll, action_space=action_space) for unroll in unrolls]
    if not mask_parts:
        raise RuntimeError("runtime learner batch requires at least one legal-action payload")
    return LegalActionBatch.from_mask(np.concatenate(mask_parts, axis=1), action_space=int(action_space))


def concatenate_legal_actions(unrolls: Sequence[Any], *, action_space: int) -> LegalActionBatch:
    saw_packed = False
    saw_mask = False

    for unroll in unrolls:
        if _unroll_legal_actions_are_packed(unroll):
            saw_packed = True
        else:
            saw_mask = True

    if saw_packed and not saw_mask:
        return _concatenate_ordered_packed_legal_actions(unrolls, action_space=action_space)
    return _concatenate_mask_legal_actions(unrolls, action_space=action_space)


def require_ids_offsets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    if batch.ids_offsets is None:
        raise RuntimeError("QueueRuntime requires ids_offsets legality batches")
    legal_ids, legal_offsets = batch.ids_offsets
    return np.asarray(legal_ids, dtype=np.uint32), np.asarray(legal_offsets, dtype=np.uint32)


def optional_legal_action_meta(batch: Any) -> np.ndarray | None:
    if batch.legal_action_meta is None:
        return None
    return np.asarray(batch.legal_action_meta, dtype=np.uint16)


def require_mask(batch: Any) -> np.ndarray:
    if batch.mask is None:
        raise RuntimeError("QueueRuntime expected dense mask legality for this actor batch")
    return np.asarray(batch.mask, dtype=np.bool_)


def concatenate_batch_legal_actions(
    batches: Sequence[Any],
    *,
    action_space: int,
) -> LegalActionBatch | None:
    if not batches:
        return None
    if all(batch.mask is not None for batch in batches):
        masks = [np.asarray(batch.mask, dtype=np.bool_) for batch in batches]
        return LegalActionBatch.from_mask(
            np.expand_dims(np.concatenate(masks, axis=0), axis=0),
            action_space=int(action_space),
        )
    if all(batch.ids_offsets is not None for batch in batches):
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets = [np.array([0], dtype=np.uint32)]
        for batch in batches:
            legal_ids, legal_offsets = require_ids_offsets(batch)
            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
            legal_action_meta = optional_legal_action_meta(batch)
            if legal_action_meta is not None:
                packed_meta.append(np.asarray(legal_action_meta, dtype=np.uint16))
            packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
        return LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(action_space),
        )
    return None


def slice_packed_rows(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ids: list[np.ndarray] = []
    offsets = [0]
    for row_index in row_indices.tolist():
        start = int(legal_offsets[int(row_index)])
        stop = int(legal_offsets[int(row_index) + 1])
        row_ids = np.asarray(legal_ids[start:stop], dtype=np.uint32)
        selected_ids.append(row_ids)
        offsets.append(offsets[-1] + int(row_ids.size))
    return (
        np.concatenate(selected_ids, axis=0) if selected_ids else np.zeros((0,), dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
    )


def slice_packed_rows_with_meta(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    *,
    legal_action_meta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    subset_ids, subset_offsets = slice_packed_rows(legal_ids, legal_offsets, row_indices)
    subset_meta = None
    if legal_action_meta is not None:
        selected_meta: list[np.ndarray] = []
        for row_index in row_indices.tolist():
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            selected_meta.append(np.asarray(legal_action_meta[start:stop], dtype=np.uint16))
        subset_meta = (
            np.concatenate(selected_meta, axis=0)
            if selected_meta
            else np.zeros((0, legal_action_meta.shape[1]), dtype=np.uint16)
        )
    return subset_ids, subset_offsets, subset_meta


def structured_legal_batch_from_mask(legal_mask: np.ndarray, row_indices: np.ndarray) -> LegalActionBatch:
    row_mask = np.asarray(legal_mask[row_indices], dtype=np.bool_)
    return LegalActionBatch.from_mask(np.expand_dims(row_mask, axis=0))


def structured_legal_batch_from_packed(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    legal_action_meta: np.ndarray | None = None,
) -> LegalActionBatch:
    subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
        legal_ids,
        legal_offsets,
        row_indices,
        legal_action_meta=legal_action_meta,
    )
    return LegalActionBatch.from_packed(subset_ids, subset_offsets, meta=subset_meta)
