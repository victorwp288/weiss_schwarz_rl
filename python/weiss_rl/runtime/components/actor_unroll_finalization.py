"""Final RuntimeUnroll assembly for generic actor collection."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from weiss_rl.runtime.components.bootstrap import bootstrap_fields_from_batch, collector_bootstrap_fields_for_actor
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.collector_unroll_storage import build_collector_runtime_unroll
from weiss_rl.runtime.components.counters import merge_simulator_timing_counters
from weiss_rl.runtime.components.types import RuntimeUnroll


@dataclass(frozen=True, slots=True)
class ActorUnrollFinalizationInputs:
    actor: Any
    batch: Any
    state: CollectorUnrollState
    action_dim: int
    started_at: float
    actor_behavior_values_required: bool
    actor_amp_enabled: bool
    bootstrap_device: Any


@dataclass(frozen=True, slots=True)
class ActorUnrollFinalizationCallbacks:
    actor_inference_model: Callable[[Any], Any]


def generic_actor_bootstrap_fields(
    *,
    actor: Any,
    batch: Any,
    values_required: bool,
    actor_inference_model: Callable[[Any], Any],
    bootstrap_device: Any,
    actor_amp_enabled: bool,
    counters: dict[str, int],
) -> Any:
    if bool(values_required):
        return collector_bootstrap_fields_for_actor(
            batch=batch,
            actor=actor,
            actor_model=actor_inference_model(actor),
            bootstrap_device=bootstrap_device,
            actor_amp_enabled=actor_amp_enabled,
            values_required=True,
            counters=counters,
        )
    return bootstrap_fields_from_batch(batch)


def finalize_generic_actor_unroll(
    *,
    inputs: ActorUnrollFinalizationInputs,
    callbacks: ActorUnrollFinalizationCallbacks,
) -> RuntimeUnroll:
    actor = inputs.actor
    state = inputs.state
    actor.current_batch = inputs.batch
    bootstrap = generic_actor_bootstrap_fields(
        actor=actor,
        batch=inputs.batch,
        values_required=inputs.actor_behavior_values_required,
        actor_inference_model=callbacks.actor_inference_model,
        bootstrap_device=inputs.bootstrap_device,
        actor_amp_enabled=inputs.actor_amp_enabled,
        counters=state.counters,
    )
    unroll = build_collector_runtime_unroll(
        actor_id=actor.actor_id,
        unroll_seq=actor.next_unroll_seq,
        behavior_policy_version=actor.snapshot_version,
        layout_name=actor.layout_name,
        action_dim=int(inputs.action_dim),
        obs=state.obs,
        actions=state.actions,
        rewards=state.rewards,
        terminated=state.terminated,
        truncated=state.truncated,
        to_play_seat=state.to_play_seat,
        behavior_logp=state.behavior_logp,
        values=state.values,
        packed_ids=state.packed_ids,
        packed_offsets=state.packed_offsets,
        packed_meta=state.packed_meta,
        mask_steps=state.mask_steps,
        bootstrap_obs=bootstrap.obs,
        bootstrap_actor=bootstrap.actor,
        bootstrap_value=bootstrap.value,
        initial_hidden_state=state.initial_hidden_state,
        final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
        episode_seed=state.episode_seed,
        policy_train_mask=state.policy_train_mask,
        opponent_context_index=state.opponent_context_index,
        teacher_family=state.teacher_family,
        teacher_slot=state.teacher_slot,
        teacher_move_source=state.teacher_move_source,
        teacher_attack_type=state.teacher_attack_type,
        teacher_action=state.teacher_action,
        teacher_valid=state.teacher_valid,
        trajectory_retention_valid=state.trajectory_retention_valid,
        counters=state.counters,
        copy_counters=False,
    )
    merge_simulator_timing_counters(state.counters, actor.env)
    state.counters["collect_actor_unroll_ms"] += int((time.perf_counter() - inputs.started_at) * 1000.0)
    actor.next_unroll_seq += 1
    return unroll


__all__ = [
    "ActorUnrollFinalizationCallbacks",
    "ActorUnrollFinalizationInputs",
    "finalize_generic_actor_unroll",
    "generic_actor_bootstrap_fields",
]
