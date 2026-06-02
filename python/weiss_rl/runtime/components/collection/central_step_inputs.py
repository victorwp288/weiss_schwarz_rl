"""Per-step input preparation for central actor collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.opponent_context import opponent_context_indices_for_model

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class CentralStepInputs(NamedTuple):
    batches: list[Any]
    obs_storage_steps: list[np.ndarray]
    obs_steps: list[np.ndarray]
    actor_steps: list[np.ndarray]


def prepare_central_step_inputs(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    states_by_actor: Mapping[int, CollectorUnrollState],
    step_index: int,
    batch_size: int,
    filter_action_surface_for_batch: Callable[..., Any],
) -> CentralStepInputs:
    filtered_batches = [
        filter_action_surface_for_batch(
            batch,
            counters=states_by_actor[int(actor.actor_id)].counters,
            action_sequence_state=states_by_actor[int(actor.actor_id)].action_sequence_state,
        )
        for actor, batch in zip(actors, batches, strict=True)
    ]
    obs_storage_steps = [np.array(batch.obs, copy=True) for batch in filtered_batches]
    obs_steps = [np.array(batch.obs, dtype=np.float32, copy=True) for batch in filtered_batches]
    actor_steps = [np.array(batch.actor, dtype=np.int64, copy=True) for batch in filtered_batches]
    for actor in actors:
        states_by_actor[int(actor.actor_id)].opponent_context_index[step_index] = opponent_context_indices_for_model(
            actor.model,
            cast(Sequence[object], actor.opponent_policy_id_by_env),
            batch_size=batch_size,
        )
    return CentralStepInputs(filtered_batches, obs_storage_steps, obs_steps, actor_steps)
