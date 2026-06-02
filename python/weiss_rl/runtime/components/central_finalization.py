"""Bootstrap and final unroll assembly for central actor collection."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.bootstrap import add_shared_elapsed_ms, bootstrap_fields_from_batches
from weiss_rl.runtime.components.central_unroll_assembly import (
    build_central_runtime_unrolls as build_central_runtime_unrolls,
)
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def compute_central_bootstrap_values(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    states_by_actor: Mapping[int, CollectorUnrollState],
    batch_size: int,
    action_dim: int,
    structured_central_packed: bool,
    values_required: bool,
    central_value_actor_rows: Callable[..., None],
    central_forward_all_rows: Callable[..., None],
    overwrite_central_outputs_with_configured_opponents: Callable[..., None],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    bootstrap_obs_steps, bootstrap_actor_steps, bootstrap_values = bootstrap_fields_from_batches(list(batches))
    if not bool(values_required):
        return bootstrap_obs_steps, bootstrap_actor_steps, bootstrap_values

    bootstrap_started = time.perf_counter()
    if structured_central_packed:
        central_value_actor_rows(
            actors=actors,
            obs_steps=bootstrap_obs_steps,
            actor_steps=bootstrap_actor_steps,
            row_indices_by_actor=[np.arange(int(batch_size), dtype=np.int64) for _ in actors],
            values_outs=bootstrap_values,
        )
    else:
        central_forward_all_rows(
            actors=actors,
            batches=batches,
            obs_steps=bootstrap_obs_steps,
            actor_steps=bootstrap_actor_steps,
            logits_outs=[np.empty((int(batch_size), int(action_dim)), dtype=np.float32) for _ in actors],
            values_outs=bootstrap_values,
        )
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="actor_bootstrap_ms",
        started_at=bootstrap_started,
        divisor=len(actors),
    )

    if structured_central_packed:
        return bootstrap_obs_steps, bootstrap_actor_steps, bootstrap_values

    overwrite_started = time.perf_counter()
    overwrite_central_outputs_with_configured_opponents(
        actors=actors,
        batches=batches,
        obs_steps=bootstrap_obs_steps,
        actor_steps=bootstrap_actor_steps,
        logits_outs=[None for _ in actors],
        values_outs=bootstrap_values,
    )
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="fixed_opponent_routing_ms",
        started_at=overwrite_started,
        divisor=len(actors),
    )
    return bootstrap_obs_steps, bootstrap_actor_steps, bootstrap_values


def finalize_central_actor_unrolls(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    states_by_actor: Mapping[int, CollectorUnrollState],
    batch_size: int,
    action_dim: int,
    structured_central_packed: bool,
    values_required: bool,
    central_started: float,
    central_value_actor_rows: Callable[..., None],
    central_forward_all_rows: Callable[..., None],
    overwrite_central_outputs_with_configured_opponents: Callable[..., None],
) -> list[RuntimeUnroll]:
    _bootstrap_obs_steps, _bootstrap_actor_steps, bootstrap_values = compute_central_bootstrap_values(
        actors=actors,
        batches=batches,
        states_by_actor=states_by_actor,
        batch_size=batch_size,
        action_dim=action_dim,
        structured_central_packed=structured_central_packed,
        values_required=values_required,
        central_value_actor_rows=central_value_actor_rows,
        central_forward_all_rows=central_forward_all_rows,
        overwrite_central_outputs_with_configured_opponents=overwrite_central_outputs_with_configured_opponents,
    )
    return build_central_runtime_unrolls(
        actors=actors,
        batches=batches,
        bootstrap_values=bootstrap_values,
        states_by_actor=states_by_actor,
        action_dim=action_dim,
        central_started=central_started,
    )
