from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.runtime.components.field_assembly import concat_batch_major_field


@dataclass(frozen=True, slots=True)
class BootstrapArraySpec:
    field_name: str
    dtype: np.dtype[Any]


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapFields:
    value: np.ndarray
    actor: np.ndarray


@dataclass(frozen=True, slots=True)
class ImpalaBootstrapFields:
    value: np.ndarray
    obs: np.ndarray
    actor: np.ndarray
    final_hidden_state: np.ndarray


def concat_bootstrap_array(unrolls: Sequence[Any], spec: BootstrapArraySpec) -> np.ndarray:
    return np.concatenate(
        [np.asarray(getattr(unroll, spec.field_name), dtype=spec.dtype) for unroll in unrolls], axis=0
    )


def runtime_bootstrap_fields(unrolls: Sequence[Any]) -> RuntimeBootstrapFields:
    return RuntimeBootstrapFields(
        value=concat_bootstrap_array(unrolls, BootstrapArraySpec("bootstrap_value", np.dtype(np.float32))),
        actor=concat_bootstrap_array(unrolls, BootstrapArraySpec("bootstrap_actor", np.dtype(np.int64))),
    )


def impala_bootstrap_fields(unrolls: Sequence[Any]) -> ImpalaBootstrapFields:
    return ImpalaBootstrapFields(
        value=concat_bootstrap_array(unrolls, BootstrapArraySpec("bootstrap_value", np.dtype(np.float32))),
        obs=concat_bootstrap_array(unrolls, BootstrapArraySpec("bootstrap_obs", np.dtype(np.float32))),
        actor=concat_bootstrap_array(unrolls, BootstrapArraySpec("bootstrap_actor", np.dtype(np.int64))),
        final_hidden_state=concat_batch_major_field(unrolls, "final_hidden_state"),
    )


def runtime_done_flags(*, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
    return np.logical_or(terminated, truncated)


def reset_before_step(done: np.ndarray) -> np.ndarray:
    reset = np.zeros_like(done, dtype=np.bool_)
    reset[1:] = done[:-1]
    return reset


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


def runtime_discounts(
    *,
    done: np.ndarray,
    to_play_seat: np.ndarray,
    bootstrap_actor: np.ndarray,
    gamma: float,
) -> np.ndarray:
    return actor_perspective_discounts(
        done=done,
        to_play_seat=to_play_seat,
        bootstrap_actor=bootstrap_actor,
        gamma=float(gamma),
    )
