"""Opponent assignment state for actor workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActorOpponentAssignmentState:
    opponent_rng: np.random.Generator | None
    current_opponent_policy_ids: np.ndarray | None
    opponent_id_by_env: np.ndarray | None


def current_opponent_policy_ids(current_opponent_policy_ids: np.ndarray | None) -> tuple[str, ...]:
    if current_opponent_policy_ids is None:
        return ()
    return tuple(str(policy_id) for policy_id in current_opponent_policy_ids.tolist())


def resample_actor_opponents(
    *,
    opponent_sampler: Any | None,
    opponent_rng: np.random.Generator | None,
    seed: int,
    actor_id: int,
    num_envs: int,
    done: np.ndarray,
    current_opponent_policy_ids: np.ndarray | None,
    opponent_id_by_env: np.ndarray | None,
    opponent_assignment_fn: Any | None,
) -> ActorOpponentAssignmentState:
    if opponent_sampler is None:
        return ActorOpponentAssignmentState(
            opponent_rng=opponent_rng,
            current_opponent_policy_ids=current_opponent_policy_ids,
            opponent_id_by_env=opponent_id_by_env,
        )
    if opponent_rng is None:
        opponent_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(actor_id), 1]))

    done_array = np.asarray(done, dtype=np.bool_)
    if done_array.shape != (int(num_envs),):
        raise ValueError(f"done must have shape ({int(num_envs)},)")

    sample_count = int(np.count_nonzero(done_array))
    if sample_count == 0:
        return ActorOpponentAssignmentState(
            opponent_rng=opponent_rng,
            current_opponent_policy_ids=current_opponent_policy_ids,
            opponent_id_by_env=opponent_id_by_env,
        )

    if current_opponent_policy_ids is None:
        current_opponent_policy_ids = np.empty((int(num_envs),), dtype=object)
    if opponent_id_by_env is None or int(opponent_id_by_env.shape[0]) != int(num_envs):
        opponent_id_by_env = np.full((int(num_envs),), "unknown", dtype=object)

    sampled_policy_ids = sampled_opponent_policy_ids(
        opponent_sampler,
        count=sample_count,
        rng=opponent_rng,
    )
    current_opponent_policy_ids[done_array] = sampled_policy_ids
    opponent_id_by_env[done_array] = sampled_policy_ids
    if opponent_assignment_fn is not None:
        opponent_assignment_fn(done_array.copy(), current_opponent_policy_ids_to_tuple(current_opponent_policy_ids))

    return ActorOpponentAssignmentState(
        opponent_rng=opponent_rng,
        current_opponent_policy_ids=current_opponent_policy_ids,
        opponent_id_by_env=opponent_id_by_env,
    )


def current_opponent_policy_ids_to_tuple(current_policy_ids: np.ndarray | None) -> tuple[str, ...]:
    return current_opponent_policy_ids(current_policy_ids)


def sampled_opponent_policy_ids(
    opponent_sampler: Any,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    sampled_policy_ids = opponent_sampler.sample(count=count, rng=rng)
    if isinstance(sampled_policy_ids, np.ndarray):
        sampled_items = sampled_policy_ids.tolist()
    else:
        sampled_items = list(sampled_policy_ids)
    if len(sampled_items) != count:
        raise ValueError(f"opponent_sampler must return {count} policy ids")
    return tuple(str(policy_id) for policy_id in sampled_items)


__all__ = [
    "ActorOpponentAssignmentState",
    "current_opponent_policy_ids",
    "resample_actor_opponents",
    "sampled_opponent_policy_ids",
]
