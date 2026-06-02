from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from weiss_rl.runtime.components.legal_batching import (
    optional_legal_action_meta,
    require_ids_offsets,
    slice_packed_rows_with_meta,
)
from weiss_rl.runtime.components.opponents.central_opponent_groups import CentralOpponentEntry


class CentralHeuristicEntryGroups(NamedTuple):
    packed: list[CentralOpponentEntry]
    mask: list[CentralOpponentEntry]


class CentralPackedHeuristicBatch(NamedTuple):
    obs_rows: np.ndarray
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    entry_counts: list[int]


def split_central_heuristic_entries(entries: Sequence[CentralOpponentEntry]) -> CentralHeuristicEntryGroups:
    packed_entries = [entry for entry in entries if entry.batch.ids_offsets is not None]
    mask_entries = [entry for entry in entries if entry.batch.ids_offsets is None]
    return CentralHeuristicEntryGroups(packed=packed_entries, mask=mask_entries)


def build_central_packed_heuristic_batch(
    entries: Sequence[CentralOpponentEntry],
    *,
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
) -> CentralPackedHeuristicBatch:
    obs_parts: list[np.ndarray] = []
    packed_ids: list[np.ndarray] = []
    packed_meta: list[np.ndarray] = []
    packed_offsets = [np.array([0], dtype=np.uint32)]
    entry_counts: list[int] = []

    for entry in entries:
        legal_ids, legal_offsets = require_ids_offsets(entry.batch)
        subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
            legal_ids,
            legal_offsets,
            entry.row_indices,
            legal_action_meta=ensure_legal_action_meta(legal_ids, optional_legal_action_meta(entry.batch)),
        )
        offset_base = int(packed_offsets[-1][-1])
        packed_ids.append(subset_ids)
        packed_offsets.append(np.asarray(subset_offsets[1:] + offset_base, dtype=np.uint32))
        if subset_meta is not None:
            packed_meta.append(subset_meta)
        obs_parts.append(np.asarray(entry.obs_step[entry.row_indices], dtype=np.int32))
        entry_counts.append(int(entry.row_indices.shape[0]))

    return CentralPackedHeuristicBatch(
        obs_rows=np.concatenate(obs_parts, axis=0) if obs_parts else np.zeros((0, 0), dtype=np.int32),
        legal_ids=np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
        legal_offsets=np.concatenate(packed_offsets, axis=0),
        legal_action_meta=np.concatenate(packed_meta, axis=0) if packed_meta else None,
        entry_counts=entry_counts,
    )


def legal_action_ids_from_mask_rows(legal_mask: np.ndarray, row_indices: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(np.uint32, copy=False)
        for row_index in row_indices.tolist()
    ]
