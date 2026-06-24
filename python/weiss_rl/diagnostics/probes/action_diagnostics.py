from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS_ACTION_ID = 51
MAIN_MOVE_BASE = 402
MAIN_MOVE_COUNT = 20


@dataclass(slots=True)
class ActionSummaryCounters:
    total_actions: int = 0
    pass_actions: int = 0
    main_move_actions: int = 0
    pass_with_nonpass_available: int = 0
    max_consecutive_main_moves: int = 0


@dataclass(slots=True)
class ActionSequenceState:
    consecutive_main_moves_by_env: np.ndarray


def make_action_sequence_state(num_envs: int) -> ActionSequenceState:
    return ActionSequenceState(consecutive_main_moves_by_env=np.zeros((int(num_envs),), dtype=np.int32))


def is_pass_action(action_id: int, *, pass_action_id: int = PASS_ACTION_ID) -> bool:
    return int(action_id) == int(pass_action_id)


def is_main_move_action(action_id: int) -> bool:
    action = int(action_id)
    return MAIN_MOVE_BASE <= action < MAIN_MOVE_BASE + MAIN_MOVE_COUNT


def update_action_summary_from_ids(
    *,
    counters: dict[str, int],
    state: ActionSequenceState,
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    pass_action_id: int = PASS_ACTION_ID,
    main_move_action: np.ndarray | None = None,
) -> None:
    action_array = np.asarray(actions, dtype=np.int64)
    legal_ids_array = np.asarray(legal_ids, dtype=np.int64)
    legal_offsets_array = np.asarray(legal_offsets, dtype=np.int64)
    if legal_offsets_array.ndim != 1 or legal_offsets_array.shape[0] != action_array.shape[0] + 1:
        raise ValueError("legal_offsets must have shape (batch + 1,)")
    main_move_flags = _coerce_optional_main_move_flags(main_move_action, expected_shape=action_array.shape)
    for env_index, action in enumerate(action_array.tolist()):
        start = int(legal_offsets_array[env_index])
        end = int(legal_offsets_array[env_index + 1])
        row_legal = legal_ids_array[start:end]
        _update_row(
            counters=counters,
            state=state,
            env_index=env_index,
            action=int(action),
            has_nonpass_legal=bool(np.any(row_legal != int(pass_action_id))),
            pass_action_id=pass_action_id,
            transition_main_move=None if main_move_flags is None else bool(main_move_flags[env_index]),
        )


def update_action_summary_from_mask(
    *,
    counters: dict[str, int],
    state: ActionSequenceState,
    actions: np.ndarray,
    legal_mask: np.ndarray,
    pass_action_id: int = PASS_ACTION_ID,
    main_move_action: np.ndarray | None = None,
) -> None:
    action_array = np.asarray(actions, dtype=np.int64)
    legal_mask_array = np.asarray(legal_mask, dtype=np.bool_)
    if legal_mask_array.shape[0] != action_array.shape[0]:
        raise ValueError("legal_mask batch dimension must match actions")
    main_move_flags = _coerce_optional_main_move_flags(main_move_action, expected_shape=action_array.shape)
    for env_index, action in enumerate(action_array.tolist()):
        row_mask = legal_mask_array[env_index]
        action_ids = np.arange(row_mask.shape[0], dtype=np.int64)
        has_nonpass_legal = bool(np.any(row_mask) and np.any(row_mask[action_ids != int(pass_action_id)]))
        _update_row(
            counters=counters,
            state=state,
            env_index=env_index,
            action=int(action),
            has_nonpass_legal=has_nonpass_legal,
            pass_action_id=pass_action_id,
            transition_main_move=None if main_move_flags is None else bool(main_move_flags[env_index]),
        )


def summarize_eval_action_counters(counters: ActionSummaryCounters) -> dict[str, int]:
    return {
        "total_actions": int(counters.total_actions),
        "pass_actions": int(counters.pass_actions),
        "main_move_actions": int(counters.main_move_actions),
        "pass_with_nonpass_available": int(counters.pass_with_nonpass_available),
        "max_consecutive_main_moves": int(counters.max_consecutive_main_moves),
    }


def update_eval_action_counters(
    *,
    counters: ActionSummaryCounters,
    state: ActionSequenceState,
    action: int,
    legal_ids: np.ndarray,
    pass_action_id: int = PASS_ACTION_ID,
) -> None:
    row_legal = np.asarray(legal_ids, dtype=np.int64)
    counters.total_actions += 1
    if is_pass_action(action, pass_action_id=pass_action_id):
        counters.pass_actions += 1
        if bool(np.any(row_legal != int(pass_action_id))):
            counters.pass_with_nonpass_available += 1
        state.consecutive_main_moves_by_env[0] = 0
        return
    if is_main_move_action(action):
        counters.main_move_actions += 1
        state.consecutive_main_moves_by_env[0] += 1
        counters.max_consecutive_main_moves = max(
            counters.max_consecutive_main_moves,
            int(state.consecutive_main_moves_by_env[0]),
        )
        return
    state.consecutive_main_moves_by_env[0] = 0


def reset_action_sequence_state(state: ActionSequenceState, done: np.ndarray) -> None:
    done_array = np.asarray(done, dtype=np.bool_)
    if done_array.ndim != 1:
        raise ValueError("done must be 1D")
    if done_array.shape[0] != state.consecutive_main_moves_by_env.shape[0]:
        raise ValueError("done must match action sequence state batch size")
    state.consecutive_main_moves_by_env[done_array] = 0


def _update_row(
    *,
    counters: dict[str, int],
    state: ActionSequenceState,
    env_index: int,
    action: int,
    has_nonpass_legal: bool,
    pass_action_id: int,
    transition_main_move: bool | None = None,
) -> None:
    counters["total_actions"] = int(counters.get("total_actions", 0)) + 1
    if is_pass_action(action, pass_action_id=pass_action_id):
        counters["pass_actions"] = int(counters.get("pass_actions", 0)) + 1
        if has_nonpass_legal:
            counters["pass_with_nonpass_available"] = int(counters.get("pass_with_nonpass_available", 0)) + 1
        state.consecutive_main_moves_by_env[env_index] = 0
        counters["max_consecutive_main_moves"] = max(
            int(counters.get("max_consecutive_main_moves", 0)),
            int(state.consecutive_main_moves_by_env[env_index]),
        )
        return

    is_main_move = is_main_move_action(action) if transition_main_move is None else bool(transition_main_move)
    if is_main_move:
        counters["main_move_actions"] = int(counters.get("main_move_actions", 0)) + 1
        state.consecutive_main_moves_by_env[env_index] += 1
        counters["max_consecutive_main_moves"] = max(
            int(counters.get("max_consecutive_main_moves", 0)),
            int(state.consecutive_main_moves_by_env[env_index]),
        )
        return

    state.consecutive_main_moves_by_env[env_index] = 0


def _coerce_optional_main_move_flags(value: np.ndarray | None, *, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    if value is None:
        return None
    flags = np.asarray(value, dtype=np.bool_)
    if flags.shape != expected_shape:
        raise ValueError(f"main_move_action must have shape {expected_shape}, got {flags.shape}")
    return flags
