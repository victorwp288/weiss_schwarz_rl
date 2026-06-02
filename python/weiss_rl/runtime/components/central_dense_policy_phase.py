"""Dense all-row central policy execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from weiss_rl.runtime.components.bootstrap import add_shared_elapsed_ms
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.policy_inference.central_policy_outputs import CentralPolicyPhaseOutputs

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def run_dense_central_policy_phase(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    obs_steps: Sequence[np.ndarray],
    actor_steps: Sequence[np.ndarray],
    states_by_actor: Mapping[int, CollectorUnrollState],
    batch_size: int,
    action_dim: int,
    record_batch_timer_ms: Callable[[str, float], None],
    central_forward_all_rows: Callable[..., None],
    overwrite_central_outputs_with_configured_opponents: Callable[..., None],
) -> CentralPolicyPhaseOutputs:
    logits_steps: list[np.ndarray | None] = [
        np.empty((int(batch_size), int(action_dim)), dtype=np.float32) for _ in actors
    ]
    value_steps = [np.empty((int(batch_size),), dtype=np.float32) for _ in actors]
    for actor, actor_step in zip(actors, actor_steps, strict=True):
        actor_step_array = np.asarray(actor_step, dtype=np.int64)
        focal_rows = np.flatnonzero(actor_step_array == actor.focal_seat_by_env)
        opponent_rows = np.flatnonzero(actor_step_array != actor.focal_seat_by_env)
        counters = states_by_actor[int(actor.actor_id)].counters
        counters["focal_row_count"] += int(focal_rows.shape[0])
        counters["opponent_row_count"] += int(opponent_rows.shape[0])

    forward_started = time.perf_counter()
    central_forward_all_rows(
        actors=actors,
        batches=batches,
        obs_steps=obs_steps,
        actor_steps=actor_steps,
        logits_outs=cast(Sequence[np.ndarray], logits_steps),
        values_outs=value_steps,
    )
    record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="actor_policy_forward_ms",
        started_at=forward_started,
        divisor=len(actors),
    )

    overwrite_started = time.perf_counter()
    overwrite_central_outputs_with_configured_opponents(
        actors=actors,
        batches=batches,
        obs_steps=obs_steps,
        actor_steps=actor_steps,
        logits_outs=logits_steps,
        values_outs=value_steps,
    )
    record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="fixed_opponent_routing_ms",
        started_at=overwrite_started,
        divisor=len(actors),
    )
    return CentralPolicyPhaseOutputs(
        logits_steps=logits_steps,
        value_steps=value_steps,
        action_steps=None,
        logp_steps=None,
    )
