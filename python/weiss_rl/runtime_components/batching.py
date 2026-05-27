from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime_components import shared as runtime_shared

_DEFAULT_ACTION_META_WIDTH = runtime_shared.DEFAULT_ACTION_META_WIDTH


def concat_time_major_field(unrolls: Sequence[Any], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    total_batch = sum(int(np.asarray(getattr(unroll, field_name)).shape[1]) for unroll in unrolls)
    result = np.empty((template.shape[0], total_batch, *template.shape[2:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        width = int(value.shape[1])
        result[:, offset : offset + width, ...] = value
        offset += width
    return result


def concat_optional_time_major_field(
    unrolls: Sequence[Any],
    field_name: str,
    *,
    missing_fill_value: Any,
) -> np.ndarray | None:
    present_values = [
        getattr(unroll, field_name, None) for unroll in unrolls if getattr(unroll, field_name, None) is not None
    ]
    if not present_values:
        return None
    template = np.asarray(present_values[0])
    total_batch = sum(int(np.asarray(unroll.obs).shape[1]) for unroll in unrolls)
    result = np.empty((template.shape[0], total_batch, *template.shape[2:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        raw_value = getattr(unroll, field_name, None)
        if raw_value is None:
            obs = np.asarray(unroll.obs)
            value = np.full((obs.shape[0], obs.shape[1], *template.shape[2:]), missing_fill_value, dtype=template.dtype)
        else:
            value = np.asarray(raw_value, dtype=template.dtype)
        width = int(value.shape[1])
        result[:, offset : offset + width, ...] = value
        offset += width
    return result


def concat_batch_major_field(unrolls: Sequence[Any], field_name: str) -> np.ndarray:
    if not unrolls:
        raise ValueError("unrolls must be non-empty")
    template = np.asarray(getattr(unrolls[0], field_name))
    total_batch = sum(int(np.asarray(getattr(unroll, field_name)).shape[0]) for unroll in unrolls)
    result = np.empty((total_batch, *template.shape[1:]), dtype=template.dtype)
    offset = 0
    for unroll in unrolls:
        value = np.asarray(getattr(unroll, field_name))
        width = int(value.shape[0])
        result[offset : offset + width, ...] = value
        offset += width
    return result


def gae_advantages(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: np.ndarray,
    discounts: np.ndarray,
    gae_lambda: float,
) -> np.ndarray:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    discounts_array = np.asarray(discounts, dtype=np.float32)
    bootstrap_array = np.asarray(bootstrap_value, dtype=np.float32)
    advantages = np.zeros_like(rewards_array, dtype=np.float32)
    gae = np.zeros((rewards_array.shape[1],), dtype=np.float32)
    next_values = bootstrap_array
    for timestep in range(rewards_array.shape[0] - 1, -1, -1):
        delta = rewards_array[timestep] + (discounts_array[timestep] * next_values) - values_array[timestep]
        gae = delta + (discounts_array[timestep] * float(gae_lambda) * gae)
        advantages[timestep] = gae
        next_values = values_array[timestep]
    return advantages


def actor_perspective_discounts(
    *,
    done: np.ndarray,
    to_play_seat: np.ndarray,
    bootstrap_actor: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Discounts signed to keep actor-perspective values in a zero-sum stream."""

    done_array = np.asarray(done, dtype=np.bool_)
    actor_array = np.asarray(to_play_seat, dtype=np.int64)
    bootstrap_array = np.asarray(bootstrap_actor, dtype=np.int64)
    if done_array.shape != actor_array.shape:
        raise ValueError("done and to_play_seat must have identical shapes")
    if actor_array.ndim != 2:
        raise ValueError("to_play_seat must be time-major [T, B]")
    if bootstrap_array.shape != (actor_array.shape[1],):
        raise ValueError("bootstrap_actor must have shape [B]")

    continuation_actor = np.empty_like(actor_array)
    if actor_array.shape[0] > 1:
        continuation_actor[:-1] = actor_array[1:]
    continuation_actor[-1] = bootstrap_array

    live = np.logical_not(done_array)
    valid_current_actor = (actor_array == 0) | (actor_array == 1)
    valid_continuation_actor = (continuation_actor == 0) | (continuation_actor == 1)
    if np.any(live & np.logical_not(valid_current_actor)):
        raise ValueError("live rows require to_play_seat in {0, 1}")
    if np.any(live & np.logical_not(valid_continuation_actor)):
        raise ValueError("live rows require continuation actor in {0, 1}")

    same_actor = continuation_actor == actor_array
    perspective_sign = np.where(same_actor, 1.0, -1.0).astype(np.float32)
    return live.astype(np.float32) * float(gamma) * perspective_sign


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


def build_impala_learner_batch(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    truncation_reward: float,
    truncation_bootstrap_value: bool,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
    terminal_outcome_backfill_reward: float = 0.0,
    terminal_outcome_trace_backfill_reward: float = 0.0,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    del truncation_reward, truncation_bootstrap_value
    concat_started = time.perf_counter()
    obs = concat_time_major_field(unrolls, "obs")
    actions = concat_time_major_field(unrolls, "actions")
    rewards = concat_time_major_field(unrolls, "rewards")
    terminated = concat_time_major_field(unrolls, "terminated")
    truncated = concat_time_major_field(unrolls, "truncated")
    to_play_seat = concat_time_major_field(unrolls, "to_play_seat")
    behavior_logp = concat_time_major_field(unrolls, "behavior_logp")
    behavior_values = concat_time_major_field(unrolls, "values")
    bootstrap_value = np.concatenate(
        [np.asarray(unroll.bootstrap_value, dtype=np.float32) for unroll in unrolls], axis=0
    )
    initial_hidden_state = concat_batch_major_field(unrolls, "initial_hidden_state")
    bootstrap_obs = np.concatenate([np.asarray(unroll.bootstrap_obs, dtype=np.float32) for unroll in unrolls], axis=0)
    bootstrap_actor = np.concatenate([np.asarray(unroll.bootstrap_actor, dtype=np.int64) for unroll in unrolls], axis=0)
    final_hidden_state = concat_batch_major_field(unrolls, "final_hidden_state")
    policy_train_mask = concat_time_major_field(unrolls, "policy_train_mask")
    opponent_context_index = concat_optional_time_major_field(
        unrolls,
        "opponent_context_index",
        missing_fill_value=0,
    )
    teacher_family = concat_optional_time_major_field(unrolls, "teacher_family", missing_fill_value=-1)
    teacher_slot = concat_optional_time_major_field(unrolls, "teacher_slot", missing_fill_value=-1)
    teacher_move_source = concat_optional_time_major_field(unrolls, "teacher_move_source", missing_fill_value=-1)
    teacher_attack_type = concat_optional_time_major_field(unrolls, "teacher_attack_type", missing_fill_value=-1)
    teacher_action = concat_optional_time_major_field(unrolls, "teacher_action", missing_fill_value=-1)
    teacher_valid = concat_optional_time_major_field(unrolls, "teacher_valid", missing_fill_value=False)
    trajectory_retention_valid = concat_optional_time_major_field(
        unrolls,
        "trajectory_retention_valid",
        missing_fill_value=False,
    )
    legal_actions = concatenate_legal_actions(unrolls, action_space=int(action_dim))
    if record_batch_timer_ms is not None:
        record_batch_timer_ms("legal_concatenation", time.perf_counter() - concat_started)
    legal_mask = None if legal_actions.mask is None else legal_actions.mask
    done = np.logical_or(terminated, truncated)
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
    reset_before_step = np.zeros_like(done, dtype=np.bool_)
    reset_before_step[1:] = done[:-1]
    # Runtime bootstrap observations are captured after resetting done rows. Until
    # pre-reset timeout values are carried per transition, bootstrapping through
    # any reset would leak value from the next episode.
    discounts = actor_perspective_discounts(
        done=done,
        to_play_seat=to_play_seat,
        bootstrap_actor=bootstrap_actor,
        gamma=float(gamma),
    )

    return {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_mask": legal_mask,
        "legal_action_meta": legal_actions.meta,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "bootstrap_obs": bootstrap_obs,
        "bootstrap_actor": bootstrap_actor,
        "final_hidden_state": final_hidden_state,
        "rewards": rewards,
        "discounts": discounts,
        "reset_before_step": reset_before_step,
        "behavior_logp": behavior_logp,
        "behavior_values": behavior_values,
        "bootstrap_value": bootstrap_value,
        "vtrace_rho_bar": float(vtrace_rho_bar),
        "vtrace_c_bar": float(vtrace_c_bar),
        "policy_train_mask": policy_train_mask,
        "opponent_context_index": opponent_context_index,
        "teacher_family": teacher_family,
        "teacher_slot": teacher_slot,
        "teacher_move_source": teacher_move_source,
        "teacher_attack_type": teacher_attack_type,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "trajectory_retention_valid": trajectory_retention_valid,
        "terminal_outcome_backfill_count": int(terminal_outcome_backfill_count),
        "terminal_outcome_backfill_total_micros": int(terminal_outcome_backfill_total_micros),
        "terminal_outcome_trace_backfill_count": int(terminal_outcome_trace_backfill_count),
        "terminal_outcome_trace_backfill_total_micros": int(terminal_outcome_trace_backfill_total_micros),
    }


def build_ppo_learner_batch(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    gae_lambda: float,
    truncation_reward: float,
    truncation_bootstrap_value: bool,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    del truncation_reward, truncation_bootstrap_value
    concat_started = time.perf_counter()
    obs = concat_time_major_field(unrolls, "obs")
    actions = concat_time_major_field(unrolls, "actions")
    rewards = concat_time_major_field(unrolls, "rewards")
    terminated = concat_time_major_field(unrolls, "terminated")
    truncated = concat_time_major_field(unrolls, "truncated")
    to_play_seat = concat_time_major_field(unrolls, "to_play_seat")
    old_logp = concat_time_major_field(unrolls, "behavior_logp")
    old_values = concat_time_major_field(unrolls, "values")
    initial_hidden_state = concat_batch_major_field(unrolls, "initial_hidden_state")
    policy_train_mask = concat_time_major_field(unrolls, "policy_train_mask")
    opponent_context_index = concat_optional_time_major_field(
        unrolls,
        "opponent_context_index",
        missing_fill_value=0,
    )
    teacher_family = concat_optional_time_major_field(unrolls, "teacher_family", missing_fill_value=-1)
    teacher_slot = concat_optional_time_major_field(unrolls, "teacher_slot", missing_fill_value=-1)
    teacher_move_source = concat_optional_time_major_field(unrolls, "teacher_move_source", missing_fill_value=-1)
    teacher_attack_type = concat_optional_time_major_field(unrolls, "teacher_attack_type", missing_fill_value=-1)
    teacher_action = concat_optional_time_major_field(unrolls, "teacher_action", missing_fill_value=-1)
    teacher_valid = concat_optional_time_major_field(unrolls, "teacher_valid", missing_fill_value=False)
    trajectory_retention_valid = concat_optional_time_major_field(
        unrolls,
        "trajectory_retention_valid",
        missing_fill_value=False,
    )
    legal_actions = concatenate_legal_actions(unrolls, action_space=int(action_dim))
    if record_batch_timer_ms is not None:
        record_batch_timer_ms("legal_concatenation", time.perf_counter() - concat_started)
    legal_mask = None if legal_actions.mask is None else legal_actions.mask

    done = np.logical_or(terminated, truncated)
    reset_before_step = np.zeros_like(done, dtype=np.bool_)
    reset_before_step[1:] = done[:-1]
    bootstrap_actor = np.concatenate([np.asarray(unroll.bootstrap_actor, dtype=np.int64) for unroll in unrolls], axis=0)
    discounts = actor_perspective_discounts(
        done=done,
        to_play_seat=to_play_seat,
        bootstrap_actor=bootstrap_actor,
        gamma=float(gamma),
    )

    bootstrap_value = np.concatenate(
        [np.asarray(unroll.bootstrap_value, dtype=np.float32) for unroll in unrolls], axis=0
    )
    advantages = gae_advantages(
        rewards=rewards,
        values=old_values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        gae_lambda=float(gae_lambda),
    )
    returns = advantages + old_values

    return {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_mask": legal_mask,
        "legal_action_meta": legal_actions.meta,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": rewards,
        "discounts": discounts,
        "reset_before_step": reset_before_step,
        "old_logp": old_logp,
        "old_values": old_values,
        "returns": returns,
        "advantages": advantages,
        "policy_train_mask": policy_train_mask,
        "opponent_context_index": opponent_context_index,
        "teacher_family": teacher_family,
        "teacher_slot": teacher_slot,
        "teacher_move_source": teacher_move_source,
        "teacher_attack_type": teacher_attack_type,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "trajectory_retention_valid": trajectory_retention_valid,
    }


def infer_packed_meta_width(unrolls: Sequence[Any]) -> int:
    for unroll in unrolls:
        if unroll.legal_actions.meta is not None:
            return int(np.asarray(unroll.legal_actions.meta).shape[1])
    return _DEFAULT_ACTION_META_WIDTH


def concatenate_legal_actions(unrolls: Sequence[Any], *, action_space: int) -> LegalActionBatch:
    packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
    mask_parts: list[np.ndarray] = []
    saw_packed = False
    saw_mask = False

    for unroll in unrolls:
        legal_actions = unroll.legal_actions
        if legal_actions.ids is not None and legal_actions.offsets is not None:
            saw_packed = True
            offset_base = int(packed_offsets[-1][-1])
            packed_offsets.append(np.asarray(legal_actions.offsets[1:] + offset_base, dtype=np.uint32))
            continue

        saw_mask = True
        mask_parts.append(
            legal_actions.to_mask(
                expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
                action_space=int(action_space),
            )
        )

    if saw_packed and not saw_mask:
        total_time_steps = int(unrolls[0].obs.shape[0])
        for unroll in unrolls[1:]:
            if int(unroll.obs.shape[0]) != total_time_steps:
                raise RuntimeError("packed legal-action concatenation requires aligned unroll lengths")
        if not all(
            unroll.legal_actions.row_count == int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls
        ):
            packed_ids: list[np.ndarray] = []
            packed_meta: list[np.ndarray] = []
            packed_offsets = [np.array([0], dtype=np.uint32)]
            any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
            for unroll in unrolls:
                legal_actions = unroll.legal_actions
                assert legal_actions.ids is not None and legal_actions.offsets is not None
                row_limit = int(unroll.obs.shape[0] * unroll.obs.shape[1])
                offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
                ids_limit = int(offsets[min(row_limit, max(offsets.size - 1, 0))])
                ids = np.asarray(legal_actions.ids[:ids_limit], dtype=np.uint32)
                offset_base = int(packed_offsets[-1][-1])
                packed_ids.append(ids)
                packed_offsets.append(np.asarray(offsets[1 : row_limit + 1] + offset_base, dtype=np.uint32))
                if any_meta and legal_actions.meta is not None:
                    packed_meta.append(np.asarray(legal_actions.meta[:ids_limit], dtype=np.uint16))
            return LegalActionBatch.from_packed(
                np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                np.concatenate(packed_offsets, axis=0),
                meta=(
                    np.concatenate(packed_meta, axis=0)
                    if packed_meta
                    else (np.zeros((0, infer_packed_meta_width(unrolls)), dtype=np.uint16) if any_meta else None)
                ),
                action_space=int(action_space),
            )

        total_ids = sum(int(np.asarray(unroll.legal_actions.ids, dtype=np.uint32).size) for unroll in unrolls)
        total_rows = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in unrolls)
        total_batch = sum(int(unroll.obs.shape[1]) for unroll in unrolls)
        ordered_packed_ids = np.empty((total_ids,), dtype=np.uint32)
        any_meta = any(unroll.legal_actions.meta is not None for unroll in unrolls)
        ordered_packed_meta = (
            np.empty((total_ids, infer_packed_meta_width(unrolls)), dtype=np.uint16)
            if any_meta and total_ids > 0
            else None
        )
        ordered_packed_offsets = np.empty((total_rows + 1,), dtype=np.uint32)
        ordered_packed_offsets[0] = 0
        ordered_widths = np.empty((total_time_steps, total_batch), dtype=np.uint32)
        batch_offset = 0
        for unroll in unrolls:
            legal_actions = unroll.legal_actions
            assert legal_actions.offsets is not None
            env_count = int(unroll.obs.shape[1])
            widths = np.diff(np.asarray(legal_actions.offsets, dtype=np.uint32)).reshape(total_time_steps, env_count)
            ordered_widths[:, batch_offset : batch_offset + env_count] = widths
            batch_offset += env_count
        ordered_packed_offsets[1:] = np.cumsum(ordered_widths.reshape(-1), dtype=np.uint64).astype(
            np.uint32, copy=False
        )
        ids_offset = 0
        for time_index in range(total_time_steps):
            for unroll in unrolls:
                legal_actions = unroll.legal_actions
                assert legal_actions.ids is not None and legal_actions.offsets is not None
                env_count = int(unroll.obs.shape[1])
                row_base = int(time_index * env_count)
                offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
                ids = np.asarray(legal_actions.ids, dtype=np.uint32)
                meta = None if legal_actions.meta is None else np.asarray(legal_actions.meta, dtype=np.uint16)
                start = int(offsets[row_base])
                end = int(offsets[row_base + env_count])
                width = end - start
                if width > 0:
                    ordered_packed_ids[ids_offset : ids_offset + width] = ids[start:end]
                    if ordered_packed_meta is not None:
                        if meta is None:
                            ordered_packed_meta[ids_offset : ids_offset + width] = np.iinfo(np.uint16).max
                        else:
                            ordered_packed_meta[ids_offset : ids_offset + width] = meta[start:end]
                ids_offset += width

        return LegalActionBatch.from_packed(
            ordered_packed_ids[:ids_offset],
            ordered_packed_offsets,
            meta=None if ordered_packed_meta is None else ordered_packed_meta[:ids_offset],
            action_space=int(action_space),
        )

    if saw_packed:
        mask_parts = [
            unroll.legal_actions.to_mask(
                expected_shape=(int(unroll.obs.shape[0]), int(unroll.obs.shape[1])),
                action_space=int(action_space),
            )
            for unroll in unrolls
        ]

    if not mask_parts:
        raise RuntimeError("runtime learner batch requires at least one legal-action payload")
    return LegalActionBatch.from_mask(np.concatenate(mask_parts, axis=1), action_space=int(action_space))


def require_ids_offsets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    if batch.ids_offsets is None:
        raise RuntimeError("QueueRuntime requires ids_offsets legality batches")
    legal_ids, legal_offsets = batch.ids_offsets
    return np.asarray(legal_ids, dtype=np.uint32), np.asarray(legal_offsets, dtype=np.uint32)


def optional_legal_action_meta(batch: Any) -> np.ndarray | None:
    if batch.legal_action_meta is None:
        return None
    return np.asarray(batch.legal_action_meta, dtype=np.uint16)


def require_mask(batch: Any) -> np.ndarray:
    if batch.mask is None:
        raise RuntimeError("QueueRuntime expected dense mask legality for this actor batch")
    return np.asarray(batch.mask, dtype=np.bool_)


def concatenate_batch_legal_actions(
    batches: Sequence[Any],
    *,
    action_space: int,
) -> LegalActionBatch | None:
    if not batches:
        return None
    if all(batch.mask is not None for batch in batches):
        masks = [np.asarray(batch.mask, dtype=np.bool_) for batch in batches]
        return LegalActionBatch.from_mask(
            np.expand_dims(np.concatenate(masks, axis=0), axis=0),
            action_space=int(action_space),
        )
    if all(batch.ids_offsets is not None for batch in batches):
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets = [np.array([0], dtype=np.uint32)]
        for batch in batches:
            legal_ids, legal_offsets = require_ids_offsets(batch)
            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(np.asarray(legal_ids, dtype=np.uint32))
            legal_action_meta = optional_legal_action_meta(batch)
            if legal_action_meta is not None:
                packed_meta.append(np.asarray(legal_action_meta, dtype=np.uint16))
            packed_offsets.append(np.asarray(legal_offsets[1:] + offset_base, dtype=np.uint32))
        return LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(action_space),
        )
    return None


def slice_packed_rows(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ids: list[np.ndarray] = []
    offsets = [0]
    for row_index in row_indices.tolist():
        start = int(legal_offsets[int(row_index)])
        stop = int(legal_offsets[int(row_index) + 1])
        row_ids = np.asarray(legal_ids[start:stop], dtype=np.uint32)
        selected_ids.append(row_ids)
        offsets.append(offsets[-1] + int(row_ids.size))
    return (
        np.concatenate(selected_ids, axis=0) if selected_ids else np.zeros((0,), dtype=np.uint32),
        np.asarray(offsets, dtype=np.uint32),
    )


def slice_packed_rows_with_meta(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    *,
    legal_action_meta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    subset_ids, subset_offsets = slice_packed_rows(legal_ids, legal_offsets, row_indices)
    subset_meta = None
    if legal_action_meta is not None:
        selected_meta: list[np.ndarray] = []
        for row_index in row_indices.tolist():
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            selected_meta.append(np.asarray(legal_action_meta[start:stop], dtype=np.uint16))
        subset_meta = (
            np.concatenate(selected_meta, axis=0)
            if selected_meta
            else np.zeros((0, legal_action_meta.shape[1]), dtype=np.uint16)
        )
    return subset_ids, subset_offsets, subset_meta


def structured_legal_batch_from_mask(legal_mask: np.ndarray, row_indices: np.ndarray) -> LegalActionBatch:
    row_mask = np.asarray(legal_mask[row_indices], dtype=np.bool_)
    return LegalActionBatch.from_mask(np.expand_dims(row_mask, axis=0))


def structured_legal_batch_from_packed(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    legal_action_meta: np.ndarray | None = None,
) -> LegalActionBatch:
    subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
        legal_ids,
        legal_offsets,
        row_indices,
        legal_action_meta=legal_action_meta,
    )
    return LegalActionBatch.from_packed(subset_ids, subset_offsets, meta=subset_meta)
