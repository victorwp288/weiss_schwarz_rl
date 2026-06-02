from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TerminalBackfillMetrics:
    outcome_count: int
    outcome_total_micros: int
    trace_count: int
    trace_total_micros: int


def apply_terminal_outcome_backfill(
    *,
    rewards: np.ndarray,
    done: np.ndarray,
    policy_train_mask: np.ndarray,
    reward: float,
) -> tuple[np.ndarray, int, int, int, int]:
    """Credit the last train row when a terminal outcome lands on a non-train row."""

    reward_value = float(reward)
    reward_array = np.asarray(rewards, dtype=np.float32)
    if reward_value <= 0.0:
        return reward_array, 0, 0, 0, 0
    done_array = np.asarray(done, dtype=np.bool_)
    train_array = np.asarray(policy_train_mask, dtype=np.bool_)
    if reward_array.shape != done_array.shape or reward_array.shape != train_array.shape:
        raise ValueError("rewards, done, and policy_train_mask must have identical time-major shapes")
    if reward_array.ndim != 2:
        raise ValueError("terminal outcome backfill expects time-major [T, B] arrays")

    shaped = reward_array.astype(np.float32, copy=True)
    last_train_step = np.full((reward_array.shape[1],), -1, dtype=np.int64)
    backfill_count = 0
    for timestep in range(int(reward_array.shape[0])):
        terminal_non_train = done_array[timestep] & ~train_array[timestep] & (reward_array[timestep] != 0.0)
        if np.any(terminal_non_train):
            for batch_index in np.flatnonzero(terminal_non_train):
                target_step = int(last_train_step[int(batch_index)])
                if target_step < 0:
                    continue
                # Simulator terminal rewards are actor-perspective. A non-train actor's
                # loss is a focal win, and vice versa.
                shaped[target_step, int(batch_index)] += np.float32(
                    -float(reward_array[timestep, int(batch_index)]) * reward_value
                )
                backfill_count += 1
        train_rows = train_array[timestep]
        if np.any(train_rows):
            last_train_step[train_rows] = int(timestep)
        terminal_rows = done_array[timestep]
        if np.any(terminal_rows):
            last_train_step[terminal_rows] = -1
    total_micros = int(round(reward_value * 1_000_000.0 * float(backfill_count)))
    return shaped, backfill_count, total_micros, 0, 0


def apply_terminal_outcome_trace_backfill(
    *,
    rewards: np.ndarray,
    done: np.ndarray,
    policy_train_mask: np.ndarray,
    reward: float,
) -> tuple[np.ndarray, int, int, int, int]:
    """Spread terminal win/loss credit to earlier train rows in the same in-batch episode suffix."""

    reward_value = float(reward)
    reward_array = np.asarray(rewards, dtype=np.float32)
    if reward_value <= 0.0:
        return reward_array, 0, 0, 0, 0
    done_array = np.asarray(done, dtype=np.bool_)
    train_array = np.asarray(policy_train_mask, dtype=np.bool_)
    if reward_array.shape != done_array.shape or reward_array.shape != train_array.shape:
        raise ValueError("rewards, done, and policy_train_mask must have identical time-major shapes")
    if reward_array.ndim != 2:
        raise ValueError("terminal outcome trace backfill expects time-major [T, B] arrays")

    shaped = reward_array.astype(np.float32, copy=True)
    train_suffixes: list[list[int]] = [[] for _ in range(int(reward_array.shape[1]))]
    credited_rows = 0
    for timestep in range(int(reward_array.shape[0])):
        for batch_index in range(int(reward_array.shape[1])):
            is_train = bool(train_array[timestep, batch_index])
            if is_train:
                train_suffixes[batch_index].append(timestep)
            if not bool(done_array[timestep, batch_index]):
                continue

            terminal_reward = float(reward_array[timestep, batch_index])
            if terminal_reward != 0.0:
                outcome_sign = 1.0 if terminal_reward > 0.0 else -1.0
                focal_outcome = outcome_sign if is_train else -outcome_sign
                target_steps = train_suffixes[batch_index]
                if is_train and target_steps and target_steps[-1] == timestep:
                    target_steps = target_steps[:-1]
                if target_steps:
                    shaped[target_steps, batch_index] += np.float32(focal_outcome * reward_value)
                    credited_rows += len(target_steps)
            train_suffixes[batch_index] = []

    total_micros = int(round(reward_value * 1_000_000.0 * float(credited_rows)))
    return shaped, 0, 0, credited_rows, total_micros


def apply_runtime_reward_backfills(
    *,
    rewards: np.ndarray,
    done: np.ndarray,
    policy_train_mask: np.ndarray,
    terminal_outcome_backfill_reward: float,
    terminal_outcome_trace_backfill_reward: float,
) -> tuple[np.ndarray, TerminalBackfillMetrics]:
    (
        rewards,
        terminal_outcome_backfill_count,
        terminal_outcome_backfill_total_micros,
        _,
        _,
    ) = apply_terminal_outcome_backfill(
        rewards=rewards,
        done=done,
        policy_train_mask=policy_train_mask,
        reward=float(terminal_outcome_backfill_reward),
    )
    (
        rewards,
        _,
        _,
        terminal_outcome_trace_backfill_count,
        terminal_outcome_trace_backfill_total_micros,
    ) = apply_terminal_outcome_trace_backfill(
        rewards=rewards,
        done=done,
        policy_train_mask=policy_train_mask,
        reward=float(terminal_outcome_trace_backfill_reward),
    )
    return rewards, TerminalBackfillMetrics(
        outcome_count=int(terminal_outcome_backfill_count),
        outcome_total_micros=int(terminal_outcome_backfill_total_micros),
        trace_count=int(terminal_outcome_trace_backfill_count),
        trace_total_micros=int(terminal_outcome_trace_backfill_total_micros),
    )
