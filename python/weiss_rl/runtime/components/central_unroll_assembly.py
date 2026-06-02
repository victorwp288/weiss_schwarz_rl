"""Final RuntimeUnroll assembly for central actor collection."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.collector_unroll_storage import build_collector_runtime_unroll
from weiss_rl.runtime.components.counters import merge_simulator_timing_counters
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def build_central_runtime_unrolls(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    bootstrap_values: Sequence[np.ndarray],
    states_by_actor: Mapping[int, CollectorUnrollState],
    action_dim: int,
    central_started: float,
) -> list[RuntimeUnroll]:
    unrolls: list[RuntimeUnroll] = []
    for actor, batch, bootstrap_value in zip(actors, batches, bootstrap_values, strict=True):
        state = states_by_actor[int(actor.actor_id)]
        actor.current_batch = batch
        merge_simulator_timing_counters(state.counters, actor.env)
        state.counters["collect_actor_unroll_ms"] += int(
            ((time.perf_counter() - central_started) * 1000.0) / max(len(actors), 1)
        )
        unrolls.append(
            build_collector_runtime_unroll(
                actor_id=actor.actor_id,
                unroll_seq=actor.next_unroll_seq,
                behavior_policy_version=actor.snapshot_version,
                layout_name=actor.layout_name,
                action_dim=int(action_dim),
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
                bootstrap_obs=np.asarray(batch.obs, dtype=np.float32),
                bootstrap_actor=np.asarray(batch.actor, dtype=np.int64),
                bootstrap_value=np.asarray(bootstrap_value, dtype=np.float32),
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
                copy_counters=True,
            )
        )
        actor.next_unroll_seq += 1
    return unrolls


__all__ = ["build_central_runtime_unrolls"]
