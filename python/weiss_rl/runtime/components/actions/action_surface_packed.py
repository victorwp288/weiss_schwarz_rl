"""Packed legality result helpers for runtime action-surface guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PackedActionSurfaceFilterResult:
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray | None
    filtered_rows: int
    filtered_actions: int


def empty_filter_result() -> PackedActionSurfaceFilterResult:
    return PackedActionSurfaceFilterResult(
        legal_ids=np.zeros((0,), dtype=np.uint32),
        legal_offsets=np.zeros((1,), dtype=np.uint32),
        legal_action_meta=None,
        filtered_rows=0,
        filtered_actions=0,
    )


def unchanged_filter_result(
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None,
) -> PackedActionSurfaceFilterResult:
    return PackedActionSurfaceFilterResult(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        filtered_rows=0,
        filtered_actions=0,
    )


def changed_filter_result(
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray,
    filtered_ids: list[np.ndarray],
    filtered_meta: list[np.ndarray],
    filtered_offsets: np.ndarray,
    filtered_rows: int,
    filtered_actions: int,
) -> PackedActionSurfaceFilterResult:
    return PackedActionSurfaceFilterResult(
        legal_ids=(
            np.concatenate(filtered_ids, axis=0).astype(legal_ids.dtype, copy=False)
            if filtered_ids
            else np.zeros((0,), dtype=legal_ids.dtype)
        ),
        legal_offsets=filtered_offsets.astype(legal_offsets.dtype, copy=False),
        legal_action_meta=(
            np.concatenate(filtered_meta, axis=0).astype(legal_action_meta.dtype, copy=False)
            if filtered_meta
            else np.zeros((0, legal_action_meta.shape[1]), dtype=legal_action_meta.dtype)
        ),
        filtered_rows=filtered_rows,
        filtered_actions=filtered_actions,
    )


__all__ = [
    "PackedActionSurfaceFilterResult",
    "changed_filter_result",
    "empty_filter_result",
    "unchanged_filter_result",
]
