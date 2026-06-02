"""Deterministic policy-logit writing helpers for queue runtime."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def write_deterministic_logits(
    *,
    logits_out: np.ndarray | None,
    row_indices: np.ndarray,
    chosen_actions: np.ndarray,
    legal_action_ids: Sequence[np.ndarray],
    action_dim: int,
) -> None:
    """Write deterministic logits for rows with unpacked legal-action ids."""

    if logits_out is None:
        return
    for row_index, chosen_action, legal_ids in zip(
        row_indices.tolist(),
        np.asarray(chosen_actions, dtype=np.int64).tolist(),
        legal_action_ids,
        strict=True,
    ):
        row_logits = np.full((int(action_dim),), -1.0e9, dtype=np.float32)
        legal_ids_np = np.asarray(legal_ids, dtype=np.int64)
        if legal_ids_np.size:
            row_logits[legal_ids_np] = -100.0
        row_logits[int(chosen_action)] = 0.0
        logits_out[int(row_index)] = row_logits


def write_deterministic_logits_from_packed(
    *,
    logits_out: np.ndarray | None,
    row_indices: np.ndarray,
    chosen_actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
) -> None:
    """Write deterministic logits for rows backed by packed legal ids and offsets."""

    if logits_out is None:
        return
    row_indices_array = np.asarray(row_indices, dtype=np.int64)
    chosen_actions_array = np.asarray(chosen_actions, dtype=np.int64)
    for row_index, chosen_action in zip(row_indices_array, chosen_actions_array, strict=True):
        row_logits = logits_out[int(row_index)]
        row_logits.fill(-1.0e9)
        start = int(legal_offsets[int(row_index)])
        stop = int(legal_offsets[int(row_index) + 1])
        if stop > start:
            row_logits[np.asarray(legal_ids[start:stop], dtype=np.int64)] = -100.0
        row_logits[int(chosen_action)] = 0.0
