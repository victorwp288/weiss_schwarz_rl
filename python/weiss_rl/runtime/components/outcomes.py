"""Runtime opponent outcome bookkeeping helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from weiss_rl.eval.harness import game_result_from_step
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

OUTCOME_COUNTER_PREFIX = "outcome_v1|"


def outcome_counter_key(opponent_policy_id: str, outcome: str) -> str:
    normalized_outcome = str(outcome).strip().lower()
    if normalized_outcome not in {"w", "l", "d", "t"}:
        raise ValueError(f"outcome must be one of w/l/d/t, got {outcome!r}")
    normalized_policy_id = str(opponent_policy_id).strip()
    if not normalized_policy_id:
        raise ValueError("opponent_policy_id must be non-empty")
    return f"{OUTCOME_COUNTER_PREFIX}{normalized_outcome}|{normalized_policy_id}"


def parse_outcome_counter_key(key: str) -> tuple[str, str] | None:
    text = str(key)
    if not text.startswith(OUTCOME_COUNTER_PREFIX):
        return None
    remainder = text[len(OUTCOME_COUNTER_PREFIX) :]
    outcome, separator, opponent_policy_id = remainder.partition("|")
    if separator != "|" or outcome not in {"w", "l", "d", "t"} or not opponent_policy_id:
        return None
    return opponent_policy_id, outcome


def record_outcome_counter(
    counters: dict[str, int] | None,
    *,
    opponent_policy_id: str,
    outcome: str,
) -> None:
    if counters is None:
        return
    key = outcome_counter_key(opponent_policy_id, outcome)
    counters[key] = int(counters.get(key, 0)) + 1


def apply_outcome_counters_to_tracker(*, outcome_tracker: Any, counters: dict[str, int] | None) -> int:
    if counters is None:
        return 0
    applied = 0
    for key, raw_count in counters.items():
        parsed = parse_outcome_counter_key(str(key))
        if parsed is None:
            continue
        opponent_policy_id, outcome = parsed
        count = int(raw_count)
        if count <= 0:
            continue
        for _ in range(count):
            outcome_tracker.update(opponent_policy_id, outcome)
        applied += count
    return applied


def update_outcomes(
    *,
    outcome_tracker: Any,
    opponent_policy_id_by_env: np.ndarray,
    focal_seat_by_env: np.ndarray,
    acting_seat: np.ndarray,
    terminal_batch: Any,
    done: np.ndarray,
    mirror_policy_id: str = MIRROR_OPPONENT_POLICY_ID,
    game_result_from_step_fn: Callable[..., Any] = game_result_from_step,
    counters: dict[str, int] | None = None,
) -> None:
    if not np.any(done):
        return
    for env_index in np.flatnonzero(done):
        opponent_policy_id = str(opponent_policy_id_by_env[env_index])
        if opponent_policy_id == mirror_policy_id:
            continue
        result = game_result_from_step_fn(
            terminal_batch,
            env_index=int(env_index),
            acting_seat=int(acting_seat[int(env_index)]),
        )
        focal_seat = int(focal_seat_by_env[int(env_index)])
        if result.truncated:
            outcome = "t"
        elif result.winner_seat is None:
            outcome = "d"
        elif int(result.winner_seat) == focal_seat:
            outcome = "w"
        else:
            outcome = "l"
        outcome_tracker.update(opponent_policy_id, outcome)
        record_outcome_counter(counters, opponent_policy_id=opponent_policy_id, outcome=outcome)


def update_outcomes_from_transition_arrays(
    *,
    outcome_tracker: Any,
    opponent_policy_id_by_env: np.ndarray,
    focal_seat_by_env: np.ndarray,
    acting_seat: np.ndarray,
    rewards: np.ndarray,
    truncated: np.ndarray,
    done: np.ndarray,
    mirror_policy_id: str = MIRROR_OPPONENT_POLICY_ID,
    counters: dict[str, int] | None = None,
) -> None:
    if not np.any(done):
        return
    acting_seat_array = np.asarray(acting_seat, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float32)
    truncated_array = np.asarray(truncated, dtype=np.bool_)
    done_array = np.asarray(done, dtype=np.bool_)
    for env_index in np.flatnonzero(done_array):
        opponent_policy_id = str(opponent_policy_id_by_env[env_index])
        if opponent_policy_id == mirror_policy_id:
            continue
        focal_seat = int(focal_seat_by_env[int(env_index)])
        if bool(truncated_array[int(env_index)]):
            outcome = "t"
        else:
            reward_value = float(reward_array[int(env_index)])
            if reward_value == 0.0:
                outcome = "d"
            else:
                winner_seat = (
                    int(acting_seat_array[int(env_index)])
                    if reward_value > 0.0
                    else 1 - int(acting_seat_array[int(env_index)])
                )
                outcome = "w" if winner_seat == focal_seat else "l"
        outcome_tracker.update(opponent_policy_id, outcome)
        record_outcome_counter(counters, opponent_policy_id=opponent_policy_id, outcome=outcome)
