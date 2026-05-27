"""Typed legality payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LegalActionBatch:
    """One legality payload with either dense masks or packed ids/offsets."""

    mask: np.ndarray | None = None
    ids: np.ndarray | None = None
    meta: np.ndarray | None = None
    offsets: np.ndarray | None = None
    action_space: int | None = None

    def __post_init__(self) -> None:
        has_mask = self.mask is not None
        has_packed = self.ids is not None or self.offsets is not None
        if self.meta is not None and not has_packed:
            raise ValueError("packed legal action meta requires packed ids/offsets")
        if has_mask == has_packed:
            raise ValueError("LegalActionBatch must contain exactly one representation")
        if has_mask:
            mask = np.asarray(self.mask, dtype=np.bool_)
            if mask.ndim != 3:
                raise ValueError(f"legal mask must be 3D (time, batch, action), got {mask.shape}")
            object.__setattr__(self, "mask", mask)
            mask_action_space = int(mask.shape[-1])
            if self.action_space is None:
                object.__setattr__(self, "action_space", mask_action_space)
            elif int(self.action_space) != mask_action_space:
                raise ValueError(
                    f"legal mask action-space mismatch: expected {int(self.action_space)}, got {mask_action_space}"
                )
            return

        if self.ids is None or self.offsets is None:
            raise ValueError("packed legality requires both ids and offsets")
        ids = np.asarray(self.ids, dtype=np.uint32)
        meta = None if self.meta is None else np.asarray(self.meta, dtype=np.uint16)
        offsets = np.asarray(self.offsets, dtype=np.uint32)
        if ids.ndim != 1:
            raise ValueError(f"packed legal ids must be 1D, got {ids.shape}")
        if meta is not None:
            if meta.ndim != 2:
                raise ValueError(f"packed legal action meta must be 2D, got {meta.shape}")
            if int(meta.shape[0]) != int(ids.shape[0]):
                raise ValueError(
                    "packed legal action meta must align 1:1 with packed legal ids: "
                    f"expected first dim {ids.shape[0]}, got {meta.shape[0]}"
                )
        if offsets.ndim != 1:
            raise ValueError(f"packed legal offsets must be 1D, got {offsets.shape}")
        if offsets.size < 1:
            raise ValueError("packed legal offsets must contain at least one element")
        if int(offsets[0]) != 0:
            raise ValueError("packed legal offsets must start at 0")
        if int(offsets[-1]) != int(ids.size):
            raise ValueError("packed legal offsets must end at len(ids)")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("packed legal offsets must be non-decreasing")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "meta", meta)
        object.__setattr__(self, "offsets", offsets)
        if self.action_space is not None:
            action_space = int(self.action_space)
            if action_space <= 0:
                raise ValueError("action_space must be positive")
            if ids.size and np.any(ids >= action_space):
                raise ValueError(f"packed legal ids must be in [0, {action_space})")
            object.__setattr__(self, "action_space", action_space)

    @classmethod
    def from_mask(cls, mask: np.ndarray | Any, *, action_space: int | None = None) -> LegalActionBatch:
        return cls(mask=np.asarray(mask, dtype=np.bool_), action_space=action_space)

    @classmethod
    def from_packed(
        cls,
        ids: np.ndarray | Any,
        offsets: np.ndarray | Any,
        *,
        meta: np.ndarray | Any | None = None,
        action_space: int | None = None,
    ) -> LegalActionBatch:
        return cls(
            ids=np.asarray(ids, dtype=np.uint32),
            meta=None if meta is None else np.asarray(meta, dtype=np.uint16),
            offsets=np.asarray(offsets, dtype=np.uint32),
            action_space=action_space,
        )

    @property
    def row_count(self) -> int:
        if self.mask is not None:
            time_steps, batch_size, _action_dim = self.mask.shape
            return int(time_steps * batch_size)
        assert self.offsets is not None
        return int(self.offsets.size - 1)

    @property
    def action_dim(self) -> int | None:
        if self.mask is not None:
            return int(self.mask.shape[-1])
        return None

    def to_mask(self, *, expected_shape: tuple[int, int], action_space: int | None = None) -> np.ndarray:
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=np.bool_, copy=False)
            if mask.shape[:2] != expected_shape:
                raise ValueError(f"legal mask shape mismatch: expected {expected_shape}, got {mask.shape[:2]}")
            resolved_action_space = self.action_space if action_space is None else action_space
            expected_action_space = int(mask.shape[2] if resolved_action_space is None else resolved_action_space)
            if mask.shape[2] != expected_action_space:
                raise ValueError(
                    f"legal mask action-space mismatch: expected {expected_action_space}, got {mask.shape[2]}"
                )
            return mask
        assert self.ids is not None and self.offsets is not None
        resolved_action_space = self.action_space if action_space is None else action_space
        if resolved_action_space is None:
            raise ValueError("packed legal actions require action_space metadata to convert to a mask")
        return packed_ids_to_mask(
            self.ids,
            self.offsets,
            expected_shape=expected_shape,
            action_space=int(resolved_action_space),
        )


def packed_ids_to_mask(
    legal_ids: np.ndarray | Any,
    legal_offsets: np.ndarray | Any,
    *,
    expected_shape: tuple[int, int],
    action_space: int,
) -> np.ndarray:
    ids = np.asarray(legal_ids, dtype=np.int64)
    offsets = np.asarray(legal_offsets, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if offsets.ndim != 1:
        raise ValueError("legal_offsets must be 1D")
    rows = int(expected_shape[0] * expected_shape[1])
    if offsets.size != rows + 1:
        raise ValueError(f"legal_offsets must have length {rows + 1}, got {offsets.size}")
    if int(offsets[0]) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1]) != int(ids.size):
        raise ValueError("legal_offsets must end at len(legal_ids)")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("legal_offsets must be non-decreasing")
    if np.any(ids < 0) or np.any(ids >= int(action_space)):
        raise ValueError(f"packed legal ids must be in [0, {action_space})")

    mask = np.zeros((rows, int(action_space)), dtype=np.bool_)
    for row_index in range(rows):
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        row_ids = ids[start:end]
        if row_ids.size:
            mask[row_index, row_ids] = True
    return mask.reshape(int(expected_shape[0]), int(expected_shape[1]), int(action_space))
