"""Actor action selection for simulator legality layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.masking import MaskingAnomalyCounters, sample_actions_from_legal_ids, sample_actions_from_mask


@dataclass(frozen=True)
class ActorActionSelection:
    actions: np.ndarray
    logp: np.ndarray
    entropy: np.ndarray
    legal_ids: np.ndarray | None = None
    legal_offsets: np.ndarray | None = None
    legal_mask: np.ndarray | None = None
    replay_legal_slices: tuple[np.ndarray, ...] = ()
    unroll_legal_ids: np.ndarray | None = None
    unroll_legal_offsets: np.ndarray | None = None


def select_legal_ids_actions(
    *,
    batch: Any,
    logits: np.ndarray,
    rng: np.random.Generator,
    counters: MaskingAnomalyCounters,
    pass_action_id: int | None,
    offset_base: int,
) -> ActorActionSelection:
    legal_ids, legal_offsets = batch_legal_ids_offsets(batch)
    legal_ids_array = np.array(legal_ids, copy=True)
    legal_offsets_array = np.array(legal_offsets, copy=True)
    packed_legal_ids_array = np.array(legal_ids_array, dtype=np.int64, copy=True)
    packed_legal_offsets_array = np.array(legal_offsets_array, dtype=np.int64, copy=True)
    actions, logp, entropy = sample_actions_from_legal_ids(
        logits,
        legal_ids_array,
        legal_offsets_array,
        rng=rng,
        counters=counters,
        pass_action_id=pass_action_id,
    )

    replay_legal_slices: list[np.ndarray] = []
    for row_index in range(int(logits.shape[0])):
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        replay_legal_slices.append(np.array(legal_ids_array[start:end], dtype=np.uint16, copy=True))

    return ActorActionSelection(
        actions=actions,
        logp=logp,
        entropy=entropy,
        legal_ids=packed_legal_ids_array,
        legal_offsets=packed_legal_offsets_array,
        replay_legal_slices=tuple(replay_legal_slices),
        unroll_legal_ids=np.array(
            packed_legal_ids_prefix(legal_ids_array, legal_offsets_array), dtype=np.int32, copy=True
        ),
        unroll_legal_offsets=np.array(legal_offsets_array[1:] + int(offset_base), dtype=np.uint32, copy=True),
    )


def select_mask_actions(
    *,
    batch: Any,
    logits: np.ndarray,
    action_space: int,
    rng: np.random.Generator,
    counters: MaskingAnomalyCounters,
    pass_action_id: int | None,
) -> ActorActionSelection:
    legal_mask = batch_legal_mask(batch)
    expected_shape = (int(logits.shape[0]), int(action_space))
    if legal_mask.shape != expected_shape:
        raise ValueError(f"expected legal_mask shape (N, A)={expected_shape}")
    legal_mask_array = np.array(legal_mask, copy=True)
    actions, logp, entropy = sample_actions_from_mask(
        logits,
        legal_mask_array,
        rng=rng,
        counters=counters,
        pass_action_id=pass_action_id,
    )
    return ActorActionSelection(
        actions=actions,
        logp=logp,
        entropy=entropy,
        legal_mask=legal_mask_array,
    )


def batch_legal_mask(batch: Any) -> np.ndarray:
    if hasattr(batch, "legal_mask"):
        return np.asarray(batch.legal_mask)
    if hasattr(batch, "mask"):
        return np.asarray(batch.mask)
    if hasattr(batch, "masks"):
        return np.asarray(batch.masks)
    raise AttributeError("mask layout batch must expose .legal_mask, .mask, or .masks")


def batch_legal_ids_offsets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    ids_offsets = getattr(batch, "ids_offsets", None)
    if ids_offsets is not None:
        if hasattr(ids_offsets, "legal_ids") and hasattr(ids_offsets, "offsets"):
            legal_ids = ids_offsets.legal_ids
            legal_offsets = ids_offsets.offsets
        else:
            legal_ids, legal_offsets = ids_offsets
        return np.asarray(legal_ids), np.asarray(legal_offsets)

    if hasattr(batch, "legal_ids") and hasattr(batch, "legal_offsets"):
        return np.asarray(batch.legal_ids), np.asarray(batch.legal_offsets)

    raise AttributeError("ids_offsets layout batch must expose .ids_offsets or (.legal_ids, .legal_offsets)")


def packed_legal_ids_prefix(legal_ids: np.ndarray, legal_offsets: np.ndarray) -> np.ndarray:
    used = 0 if legal_offsets.size == 0 else int(legal_offsets[-1])
    if used < 0 or used > legal_ids.shape[0]:
        raise ValueError(f"legal_ids prefix out of bounds: used={used}, capacity={legal_ids.shape[0]}")
    return legal_ids[:used]


__all__ = [
    "ActorActionSelection",
    "batch_legal_ids_offsets",
    "batch_legal_mask",
    "packed_legal_ids_prefix",
    "select_legal_ids_actions",
    "select_mask_actions",
]
