"""Learner-side reward shaping helpers for runtime collectors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def pass_penalty_ignored_alternative_family_ids(family_index: Mapping[str, int] | None) -> tuple[int, ...]:
    """Family ids that should not make pass look like a missed tactical action."""

    if not family_index:
        return ()
    main_move_id = int(family_index.get("main_move", -1))
    return (main_move_id,) if main_move_id >= 0 else ()


def pass_with_nonpass_penalty_mask_from_ids(
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int,
    legal_action_meta: np.ndarray | None = None,
    ignored_alternative_family_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Rows where the sampled action is pass despite a productive non-pass legal action."""

    action_array = np.asarray(actions, dtype=np.int64)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    legal_offsets_array = np.asarray(legal_offsets, dtype=np.int64)
    if action_array.ndim != 1:
        raise ValueError("actions must be 1D")
    if legal_offsets_array.ndim != 1 or legal_offsets_array.shape[0] != action_array.shape[0] + 1:
        raise ValueError("legal_offsets must have shape (batch + 1,)")
    ignored_family_ids = {
        int(family_id)
        for family_id in (() if ignored_alternative_family_ids is None else ignored_alternative_family_ids)
        if int(family_id) >= 0
    }
    legal_meta_array = None
    family_ids = None
    if ignored_family_ids:
        if legal_action_meta is None:
            ignored_family_ids = set()
        else:
            legal_meta_array = np.asarray(legal_action_meta, dtype=np.int64)
            if legal_meta_array.ndim != 2 or legal_meta_array.shape[0] != legal_ids_array.shape[0]:
                raise ValueError("legal_action_meta must have shape (num_legal, meta_width)")
            if legal_meta_array.shape[1] < 1:
                raise ValueError("legal_action_meta must include family ids in column 0")
            family_ids = legal_meta_array[:, 0]
    mask = np.zeros(action_array.shape, dtype=np.bool_)
    pass_id = int(pass_action_id)
    for row_index, action in enumerate(action_array.tolist()):
        if int(action) != pass_id:
            continue
        start = int(legal_offsets_array[row_index])
        end = int(legal_offsets_array[row_index + 1])
        row_alternatives = legal_ids_array[start:end] != pass_id
        if ignored_family_ids and family_ids is not None:
            row_families = family_ids[start:end]
            row_alternatives &= ~np.isin(row_families, list(ignored_family_ids))
        if bool(np.any(row_alternatives)):
            mask[row_index] = True
    return mask


def pass_with_nonpass_penalty_mask_from_mask(
    actions: np.ndarray,
    legal_mask: np.ndarray,
    *,
    pass_action_id: int,
) -> np.ndarray:
    """Rows where the sampled action is pass despite any other legal action."""

    action_array = np.asarray(actions, dtype=np.int64)
    legal_mask_array = np.asarray(legal_mask, dtype=np.bool_)
    if action_array.ndim != 1:
        raise ValueError("actions must be 1D")
    if legal_mask_array.ndim != 2 or legal_mask_array.shape[0] != action_array.shape[0]:
        raise ValueError("legal_mask must have shape (batch, action_space)")
    pass_id = int(pass_action_id)
    nonpass_mask = legal_mask_array.copy()
    if 0 <= pass_id < nonpass_mask.shape[1]:
        nonpass_mask[:, pass_id] = False
    has_nonpass = np.any(nonpass_mask, axis=1)
    return (action_array == pass_id) & has_nonpass


def apply_pass_with_nonpass_penalty(
    rewards: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int,
    penalty: float,
    legal_ids: np.ndarray | None = None,
    legal_offsets: np.ndarray | None = None,
    legal_action_meta: np.ndarray | None = None,
    ignored_alternative_family_ids: Iterable[int] | None = None,
    legal_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return learner rewards after subtracting the configured pass penalty.

    The simulator reward is actor-perspective and remains the source of truth;
    this helper is only for learner-side shaping during B1 training.
    """

    reward_array = np.asarray(rewards, dtype=np.float32)
    penalty_value = float(penalty)
    if penalty_value <= 0.0:
        return reward_array.astype(np.float32, copy=True), 0, 0
    if legal_mask is not None:
        penalty_mask = pass_with_nonpass_penalty_mask_from_mask(
            actions,
            legal_mask,
            pass_action_id=pass_action_id,
        )
    elif legal_ids is not None and legal_offsets is not None:
        penalty_mask = pass_with_nonpass_penalty_mask_from_ids(
            actions,
            legal_ids,
            legal_offsets,
            pass_action_id=pass_action_id,
            legal_action_meta=legal_action_meta,
            ignored_alternative_family_ids=ignored_alternative_family_ids,
        )
    else:
        raise ValueError("either legal_mask or legal_ids/legal_offsets is required")
    shaped = reward_array.astype(np.float32, copy=True)
    count = int(np.count_nonzero(penalty_mask))
    if count == 0:
        return shaped, 0, 0
    shaped[penalty_mask] -= np.float32(penalty_value)
    total_micros = int(round(float(penalty_value) * 1_000_000.0 * float(count)))
    return shaped, count, total_micros


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


def apply_collector_reward_shaping(
    rewards: np.ndarray,
    actions: np.ndarray,
    *,
    counters: dict[str, int],
    pass_action_id: int,
    pass_with_nonpass_penalty: float,
    mulligan_select_with_confirm_penalty: float,
    action_family_index: Mapping[str, int] | None = None,
    legal_ids: np.ndarray | None = None,
    legal_offsets: np.ndarray | None = None,
    legal_action_meta: np.ndarray | None = None,
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply collector-side reward shaping and update collector counters."""

    shaped, penalty_count, penalty_total_micros = apply_pass_with_nonpass_penalty(
        rewards,
        actions,
        pass_action_id=int(pass_action_id),
        penalty=float(pass_with_nonpass_penalty),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        ignored_alternative_family_ids=pass_penalty_ignored_alternative_family_ids(action_family_index),
        legal_mask=legal_mask,
    )
    counters["pass_with_nonpass_penalty_count"] += penalty_count
    counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros

    if legal_ids is None or legal_offsets is None:
        return shaped

    family_index = {} if action_family_index is None else action_family_index
    shaped, penalty_count, penalty_total_micros = apply_mulligan_select_with_confirm_penalty(
        shaped,
        actions,
        penalty=float(mulligan_select_with_confirm_penalty),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
        mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
    )
    counters["mulligan_select_with_confirm_penalty_count"] += penalty_count
    counters["mulligan_select_with_confirm_penalty_total_micros"] += penalty_total_micros
    return shaped
