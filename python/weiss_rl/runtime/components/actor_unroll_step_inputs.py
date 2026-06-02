"""Per-step input preparation for generic actor-unroll collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.runtime.components.counters import accumulate_actor_role_row_counters
from weiss_rl.runtime.components.opponent_context import opponent_context_indices_for_model


@dataclass(frozen=True, slots=True)
class ActorUnrollStepInputs:
    batch: Any
    obs_storage_step: np.ndarray
    obs_step: np.ndarray
    actor_step: np.ndarray
    focal_rows: np.ndarray


def prepare_actor_unroll_step_inputs(
    *,
    actor: Any,
    batch: Any,
    step_index: int,
    batch_size: int,
    observation_dim: int,
    opponent_context_index: np.ndarray,
    counters: dict[str, int],
    action_sequence_state: Any,
    filter_action_surface_for_batch: Callable[..., Any],
) -> ActorUnrollStepInputs:
    filtered_batch = filter_action_surface_for_batch(
        batch,
        counters=counters,
        action_sequence_state=action_sequence_state,
    )
    obs_storage_step = np.array(filtered_batch.obs, copy=True)
    obs_step = np.array(filtered_batch.obs, dtype=np.float32, copy=True)
    actor_step = np.array(filtered_batch.actor, dtype=np.int64, copy=True)
    if obs_step.shape != (int(batch_size), int(observation_dim)):
        raise RuntimeError(f"unexpected actor obs shape: {obs_step.shape}")
    if np.any((actor_step != 0) & (actor_step != 1)):
        raise RuntimeError(f"actor runtime only supports live seat rows, got {actor_step.tolist()}")

    opponent_context_index[int(step_index)] = opponent_context_indices_for_model(
        actor.model,
        actor.opponent_policy_id_by_env,
        batch_size=int(batch_size),
    )
    focal_rows = actor_step == actor.focal_seat_by_env
    accumulate_actor_role_row_counters(
        counters=counters,
        actor_step=actor_step,
        focal_seat_by_env=actor.focal_seat_by_env,
    )
    return ActorUnrollStepInputs(
        batch=filtered_batch,
        obs_storage_step=obs_storage_step,
        obs_step=obs_step,
        actor_step=actor_step,
        focal_rows=focal_rows,
    )


__all__ = [
    "ActorUnrollStepInputs",
    "prepare_actor_unroll_step_inputs",
]
