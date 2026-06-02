"""Action coercion and legality validation for decision-boundary environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from weiss_rl.envs.decision_batch import DecisionBoundaryBatch


def _coerce_actions(
    actions: Sequence[int] | np.ndarray | int,
    *,
    num_envs: int,
    action_space: int,
) -> np.ndarray:
    if isinstance(actions, np.ndarray):
        action_array = actions
    elif np.isscalar(actions):
        action_array = np.asarray([actions])
    else:
        action_array = np.asarray(list(cast(Sequence[int], actions)))

    if action_array.ndim != 1 or int(action_array.shape[0]) != num_envs:
        raise ValueError(f"actions must have shape ({num_envs},)")
    if not np.issubdtype(action_array.dtype, np.integer):
        raise TypeError("actions must be integers")

    signed = action_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("actions must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"actions must be < action_space ({action_space})")
    return signed.astype(np.uint32, copy=False)


def _validate_actions(
    actions: np.ndarray,
    batch: DecisionBoundaryBatch,
    *,
    pass_action_id: int,
) -> None:
    if batch.mask is not None:
        _validate_mask_actions(actions, batch.mask, pass_action_id=pass_action_id)
        return

    assert batch.ids_offsets is not None
    legal_ids, legal_offsets = batch.ids_offsets
    _validate_packed_actions(actions, legal_ids, legal_offsets, pass_action_id=pass_action_id)


def _validate_mask_actions(actions: np.ndarray, mask: np.ndarray, *, pass_action_id: int) -> None:
    for env_index, action in enumerate(actions.tolist()):
        legal_row = mask[env_index]
        if not bool(np.any(legal_row)):
            _require_pass_action(action, env_index, pass_action_id)
            continue
        if not bool(legal_row[action]):
            raise ValueError(f"illegal action {action} for env {env_index}")


def _validate_packed_actions(
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int,
) -> None:
    for env_index, action in enumerate(actions.tolist()):
        start = int(legal_offsets[env_index])
        end = int(legal_offsets[env_index + 1])
        if start == end:
            _require_pass_action(action, env_index, pass_action_id)
            continue

        env_legal_ids = legal_ids[start:end]
        position = int(np.searchsorted(env_legal_ids, action))
        is_legal = position < env_legal_ids.size and int(env_legal_ids[position]) == action
        if not is_legal:
            raise ValueError(f"illegal action {action} for env {env_index}")


def _require_pass_action(action: int, env_index: int, pass_action_id: int) -> None:
    if action != pass_action_id:
        raise ValueError(f"env {env_index} has no legal actions; expected pass action {pass_action_id}, got {action}")


__all__ = ["_coerce_actions", "_validate_actions"]
