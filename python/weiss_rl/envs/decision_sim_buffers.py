"""Simulator output-buffer helpers for decision-boundary environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.envs.decision_batch import CopyCasting

_COMMON_OUT_FIELDS = (
    "rewards",
    "terminated",
    "truncated",
    "actor",
    "decision_kind",
    "decision_id",
    "engine_status",
    "decision_count",
    "tick_count",
    "main_move_action",
    "main_pass_action",
    "spec_hash",
)


def _make_sim_out(weiss_sim: Any, *, class_name: str, num_envs: int) -> Any:
    out_cls = getattr(weiss_sim, class_name, None)
    if out_cls is None:
        raise RuntimeError(f"weiss_sim is missing required output buffer class {class_name}")
    return out_cls(num_envs)


def _copy_common_out_fields(*, src: Any, dst: Any, rows: np.ndarray | None = None) -> None:
    for field_name in _COMMON_OUT_FIELDS:
        if not hasattr(src, field_name) or not hasattr(dst, field_name):
            continue
        _copy_rows(dst=getattr(dst, field_name), src=getattr(src, field_name), rows=rows)


def _copy_obs_into(*, src: np.ndarray, dst: np.ndarray, rows: np.ndarray | None = None) -> None:
    if np.issubdtype(dst.dtype, np.integer):
        bounds = np.iinfo(dst.dtype)
        clipped = np.clip(src, bounds.min, bounds.max)
        _copy_rows(dst=dst, src=clipped.astype(dst.dtype, copy=False), rows=rows, casting="unsafe")
        return
    _copy_rows(dst=dst, src=src, rows=rows)


def _copy_rows(
    *,
    dst: np.ndarray,
    src: np.ndarray,
    rows: np.ndarray | None = None,
    casting: CopyCasting = "same_kind",
) -> None:
    if rows is None:
        np.copyto(dst, src, casting=casting)
        return
    dst[rows] = np.asarray(src)[rows].astype(dst.dtype, copy=False)


def _merge_packed_legality_rows(*, dst: Any, current: Any, replacement: Any, rows: np.ndarray) -> None:
    current_ids = np.asarray(current.legal_ids)
    current_offsets = np.asarray(current.legal_offsets)
    current_meta_raw = getattr(current, "legal_action_meta", None)
    current_meta = None if current_meta_raw is None else np.asarray(current_meta_raw)
    replacement_ids = np.asarray(replacement.legal_ids)
    replacement_offsets = np.asarray(replacement.legal_offsets)
    replacement_meta_raw = getattr(replacement, "legal_action_meta", None)
    replacement_meta = None if replacement_meta_raw is None else np.asarray(replacement_meta_raw)

    merged_ids_parts: list[np.ndarray] = []
    merged_meta_parts: list[np.ndarray] = []
    merged_offsets = np.zeros((rows.shape[0] + 1,), dtype=current_offsets.dtype)
    cursor = 0
    meta_template = current_meta if current_meta is not None else replacement_meta
    for row_index, replace_row in enumerate(rows.tolist()):
        if replace_row:
            row_ids = replacement_ids[int(replacement_offsets[row_index]) : int(replacement_offsets[row_index + 1])]
            row_meta = (
                None
                if replacement_meta is None
                else replacement_meta[int(replacement_offsets[row_index]) : int(replacement_offsets[row_index + 1])]
            )
        else:
            row_ids = current_ids[int(current_offsets[row_index]) : int(current_offsets[row_index + 1])]
            row_meta = (
                None
                if current_meta is None
                else current_meta[int(current_offsets[row_index]) : int(current_offsets[row_index + 1])]
            )
        row_ids = np.array(row_ids, copy=True)
        merged_ids_parts.append(row_ids)
        if meta_template is not None:
            if row_meta is None:
                row_meta = np.full(
                    (int(row_ids.size), int(meta_template.shape[1])),
                    np.iinfo(meta_template.dtype).max,
                    dtype=meta_template.dtype,
                )
            else:
                row_meta = np.array(row_meta, copy=True)
            merged_meta_parts.append(row_meta)
        cursor += int(row_ids.size)
        merged_offsets[row_index + 1] = cursor

    merged_ids = (
        np.concatenate(merged_ids_parts, axis=0).astype(current_ids.dtype, copy=False)
        if merged_ids_parts
        else np.zeros((0,), dtype=current_ids.dtype)
    )
    merged_meta = np.concatenate(merged_meta_parts, axis=0) if merged_meta_parts else None
    _write_packed_legality(
        dst=dst,
        legal_ids=merged_ids,
        legal_offsets=merged_offsets,
        legal_action_meta=merged_meta,
    )


def _write_packed_legality(
    *,
    dst: Any,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray | None = None,
) -> None:
    dst_ids = np.asarray(dst.legal_ids)
    dst_offsets = np.asarray(dst.legal_offsets)
    dst_meta_raw = getattr(dst, "legal_action_meta", None)
    dst_meta = None if dst_meta_raw is None else np.asarray(dst_meta_raw)

    if dst_offsets.shape != legal_offsets.shape:
        raise RuntimeError(
            f"packed legal_offsets shape mismatch: expected {dst_offsets.shape}, got {legal_offsets.shape}"
        )
    if legal_ids.size > dst_ids.shape[0]:
        raise RuntimeError(f"packed legal_ids buffer too small: capacity={dst_ids.shape[0]}, required={legal_ids.size}")

    np.copyto(dst_offsets, legal_offsets.astype(dst_offsets.dtype, copy=False), casting="unsafe")
    if legal_ids.size:
        np.copyto(dst_ids[: legal_ids.size], legal_ids.astype(dst_ids.dtype, copy=False), casting="unsafe")
    if legal_ids.size < dst_ids.shape[0]:
        dst_ids[legal_ids.size :] = 0
    if dst_meta is not None:
        fill_value = np.iinfo(dst_meta.dtype).max
        dst_meta[...] = fill_value
        if legal_action_meta is not None:
            meta = np.asarray(legal_action_meta, dtype=dst_meta.dtype)
            if meta.ndim != 2:
                raise RuntimeError(f"packed legal_action_meta must be 2D, got {meta.shape}")
            if int(meta.shape[0]) != int(legal_ids.size):
                raise RuntimeError(
                    "packed legal_action_meta must align with packed legal ids: "
                    f"expected first dim {legal_ids.size}, got {meta.shape[0]}"
                )
            if int(meta.shape[1]) != int(dst_meta.shape[1]):
                raise RuntimeError(
                    f"packed legal_action_meta width mismatch: expected {dst_meta.shape[1]}, got {meta.shape[1]}"
                )
            if meta.size:
                np.copyto(dst_meta[: meta.shape[0]], meta, casting="unsafe")


__all__ = [
    "_copy_common_out_fields",
    "_copy_obs_into",
    "_copy_rows",
    "_make_sim_out",
    "_merge_packed_legality_rows",
    "_write_packed_legality",
]
