"""Setup state for central actor-unroll collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.collector_state import CollectorUnrollState, allocate_collector_unroll_state
from weiss_rl.runtime.components.counters import timeout_limits_for_env

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState

TimeoutLimits = dict[str, int | None]


@dataclass(frozen=True, slots=True)
class CentralActorCollectionSetup:
    time_steps: int
    batch_size: int
    obs_dtype: np.dtype[Any]
    batches: list[DecisionBoundaryBatch]
    states_by_actor: dict[int, CollectorUnrollState]
    timeout_limits_by_actor: dict[int, TimeoutLimits]
    structured_central_packed: bool


def actors_have_single_layout(actors: Sequence[_ActorState]) -> bool:
    return len({str(actor.layout_name) for actor in actors}) <= 1


def supports_structured_central_packed(
    actors: Sequence[_ActorState],
    *,
    actor_inference_model: Callable[[_ActorState], Any],
) -> bool:
    if not actors:
        return False
    return bool(
        all(actor.layout_name == "i16_legal_ids" for actor in actors)
        and bool(getattr(actor_inference_model(actors[0]), "supports_legal_candidate_scoring", False))
    )


def build_central_actor_collection_setup(
    *,
    actors: Sequence[_ActorState],
    config: Any,
    observation_dim: int,
    trajectory_retention_enabled: bool,
    actor_inference_model: Callable[[_ActorState], Any],
    actor_timeout_limits: Callable[[Any], TimeoutLimits] = timeout_limits_for_env,
) -> CentralActorCollectionSetup:
    if not actors:
        raise ValueError("central actor collection setup requires at least one actor")

    time_steps = int(config.unroll_length)
    batch_size = int(config.envs_per_actor)
    obs_dtype = np.asarray(actors[0].current_batch.obs).dtype
    states_by_actor = {
        int(actor.actor_id): allocate_collector_unroll_state(
            time_steps=time_steps,
            batch_size=batch_size,
            observation_dim=observation_dim,
            obs_dtype=obs_dtype,
            seat_hidden=actor.seat_hidden,
            trajectory_retention_enabled=trajectory_retention_enabled,
        )
        for actor in actors
    }
    return CentralActorCollectionSetup(
        time_steps=time_steps,
        batch_size=batch_size,
        obs_dtype=obs_dtype,
        batches=[actor.current_batch for actor in actors],
        states_by_actor=states_by_actor,
        timeout_limits_by_actor={int(actor.actor_id): actor_timeout_limits(actor.env) for actor in actors},
        structured_central_packed=supports_structured_central_packed(
            actors,
            actor_inference_model=actor_inference_model,
        ),
    )


__all__ = [
    "CentralActorCollectionSetup",
    "TimeoutLimits",
    "actors_have_single_layout",
    "build_central_actor_collection_setup",
    "supports_structured_central_packed",
]
