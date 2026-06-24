"""Mulligan-select learner reward shaping."""

from __future__ import annotations

import numpy as np


def mulligan_select_with_confirm_penalty_mask_from_ids(
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    legal_action_meta: np.ndarray,
    *,
    mulligan_select_family_id: int,
    mulligan_confirm_family_id: int,
) -> np.ndarray:
    """Rows where a mulligan-select action was sampled while confirm was legal."""

    action_array = np.asarray(actions, dtype=np.int64)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    legal_offsets_array = np.asarray(legal_offsets, dtype=np.int64)
    legal_meta_array = np.asarray(legal_action_meta, dtype=np.int64)
    if action_array.ndim != 1:
        raise ValueError("actions must be 1D")
    if legal_offsets_array.ndim != 1 or legal_offsets_array.shape[0] != action_array.shape[0] + 1:
        raise ValueError("legal_offsets must have shape (batch + 1,)")
    if legal_meta_array.ndim != 2 or legal_meta_array.shape[0] != legal_ids_array.shape[0]:
        raise ValueError("legal_action_meta must have shape (num_legal, meta_width)")
    if legal_meta_array.shape[1] < 1:
        raise ValueError("legal_action_meta must include family ids in column 0")

    select_family = int(mulligan_select_family_id)
    confirm_family = int(mulligan_confirm_family_id)
    if select_family < 0 or confirm_family < 0:
        raise ValueError("mulligan family ids must be present when mulligan-select penalty is enabled")

    mask = np.zeros(action_array.shape, dtype=np.bool_)
    family_ids = legal_meta_array[:, 0]
    for row_index, action in enumerate(action_array.tolist()):
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        row_ids = legal_ids_array[start:end]
        row_families = family_ids[start:end]
        if row_ids.size == 0 or not bool(np.any(row_families == confirm_family)):
            continue
        selected_positions = row_ids == int(action)
        if bool(np.any(row_families[selected_positions] == select_family)):
            mask[row_index] = True
    return mask


def apply_mulligan_select_with_confirm_penalty(
    rewards: np.ndarray,
    actions: np.ndarray,
    *,
    penalty: float,
    legal_ids: np.ndarray | None,
    legal_offsets: np.ndarray | None,
    legal_action_meta: np.ndarray | None,
    mulligan_select_family_id: int,
    mulligan_confirm_family_id: int,
) -> tuple[np.ndarray, int, int]:
    """Return learner rewards after subtracting the configured mulligan-select penalty."""

    reward_array = np.asarray(rewards, dtype=np.float32)
    penalty_value = float(penalty)
    if penalty_value <= 0.0:
        return reward_array.astype(np.float32, copy=True), 0, 0
    if legal_ids is None or legal_offsets is None or legal_action_meta is None:
        raise ValueError("legal_ids, legal_offsets, and legal_action_meta are required")
    penalty_mask = mulligan_select_with_confirm_penalty_mask_from_ids(
        actions,
        legal_ids,
        legal_offsets,
        legal_action_meta,
        mulligan_select_family_id=int(mulligan_select_family_id),
        mulligan_confirm_family_id=int(mulligan_confirm_family_id),
    )
    shaped = reward_array.astype(np.float32, copy=True)
    count = int(np.count_nonzero(penalty_mask))
    if count == 0:
        return shaped, 0, 0
    shaped[penalty_mask] -= np.float32(penalty_value)
    total_micros = int(round(float(penalty_value) * 1_000_000.0 * float(count)))
    return shaped, count, total_micros
