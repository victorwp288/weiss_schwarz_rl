"""Packed structured central policy execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.bootstrap import add_shared_elapsed_ms
from weiss_rl.runtime.components.central_row_partitions import partition_central_actor_rows
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.legal_batching import optional_legal_action_meta, require_ids_offsets
from weiss_rl.runtime.components.policy_inference.central_policy_outputs import CentralPolicyPhaseOutputs

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def run_structured_central_policy_phase(
    *,
    actors: Sequence[_ActorState],
    batches: Sequence[Any],
    obs_steps: Sequence[np.ndarray],
    actor_steps: Sequence[np.ndarray],
    states_by_actor: Mapping[int, CollectorUnrollState],
    batch_size: int,
    opponent_heuristic_policy_ids: Sequence[str],
    fuse_mirror_policy_rows: bool,
    record_batch_timer_ms: Callable[[str, float], None],
    central_sample_policy_rows_ids: Callable[..., None],
    central_advance_actor_rows: Callable[..., None],
    should_track_heuristic_actor_hidden_state: Callable[[], bool],
    apply_opponent_rows_ids: Callable[..., None],
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
) -> CentralPolicyPhaseOutputs:
    action_steps = [np.zeros((int(batch_size),), dtype=np.int64) for _ in actors]
    logp_steps = [np.zeros((int(batch_size),), dtype=np.float32) for _ in actors]
    value_steps = [np.zeros((int(batch_size),), dtype=np.float32) for _ in actors]
    heuristic_policy_ids = tuple(str(policy_id) for policy_id in opponent_heuristic_policy_ids)
    row_partitions = partition_central_actor_rows(
        actors=actors,
        actor_steps=actor_steps,
        heuristic_policy_ids=heuristic_policy_ids,
        fuse_mirror_policy_rows=fuse_mirror_policy_rows,
    )
    for actor, partition in zip(actors, row_partitions.entries, strict=True):
        counters = states_by_actor[int(actor.actor_id)].counters
        counters["focal_row_count"] += int(partition.focal_rows.shape[0])
        counters["opponent_row_count"] += partition.opponent_row_count

    sampled_policy_rows_by_actor = row_partitions.sampled_policy_rows_by_actor
    forward_started = time.perf_counter()
    if any(rows.size > 0 for rows in sampled_policy_rows_by_actor):
        central_sample_policy_rows_ids(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            row_indices_by_actor=sampled_policy_rows_by_actor,
            values_outs=value_steps,
            actions_outs=action_steps,
            logp_outs=logp_steps,
        )
    record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="actor_policy_forward_ms",
        started_at=forward_started,
        divisor=len(actors),
    )

    overwrite_started = time.perf_counter()
    heuristic_rows_by_actor = row_partitions.heuristic_rows_by_actor
    if fuse_mirror_policy_rows and heuristic_policy_ids:
        _advance_heuristic_rows_if_needed(
            actors=actors,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            heuristic_rows_by_actor=heuristic_rows_by_actor,
            central_advance_actor_rows=central_advance_actor_rows,
            should_track_heuristic_actor_hidden_state=should_track_heuristic_actor_hidden_state,
        )
    for actor, batch, obs_step, actor_step, value_step, action_step, logp_step, heuristic_rows, residual_rows in zip(
        actors,
        batches,
        obs_steps,
        actor_steps,
        value_steps,
        action_steps,
        logp_steps,
        heuristic_rows_by_actor,
        row_partitions.residual_rows_by_actor,
        strict=True,
    ):
        legal_ids, legal_offsets = require_ids_offsets(batch)
        legal_action_meta = ensure_legal_action_meta(legal_ids, optional_legal_action_meta(batch))
        if fuse_mirror_policy_rows:
            if heuristic_rows.size > 0:
                apply_opponent_rows_ids(
                    actor=actor,
                    row_indices=heuristic_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    logits_out=None,
                    values_out=value_step,
                    actions_out=action_step,
                    logp_out=logp_step,
                    rng=actor.rng,
                    sample_actions=True,
                    heuristic_rows_hidden_already_advanced=True,
                )
            opponent_rows = residual_rows
        else:
            opponent_rows = np.flatnonzero(np.asarray(actor_step, dtype=np.int64) != actor.focal_seat_by_env)
        if opponent_rows.size > 0:
            apply_opponent_rows_ids(
                actor=actor,
                row_indices=opponent_rows,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=None,
                values_out=value_step,
                actions_out=action_step,
                logp_out=logp_step,
                rng=actor.rng,
                sample_actions=True,
            )
    record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
    add_shared_elapsed_ms(
        counters=[state.counters for state in states_by_actor.values()],
        key="fixed_opponent_routing_ms",
        started_at=overwrite_started,
        divisor=len(actors),
    )
    return CentralPolicyPhaseOutputs(
        logits_steps=[None for _ in actors],
        value_steps=value_steps,
        action_steps=action_steps,
        logp_steps=logp_steps,
    )


def _advance_heuristic_rows_if_needed(
    *,
    actors: Sequence[_ActorState],
    obs_steps: Sequence[np.ndarray],
    actor_steps: Sequence[np.ndarray],
    heuristic_rows_by_actor: Sequence[np.ndarray],
    central_advance_actor_rows: Callable[..., None],
    should_track_heuristic_actor_hidden_state: Callable[[], bool],
) -> None:
    heuristic_actors: list[_ActorState] = []
    heuristic_obs_steps: list[np.ndarray] = []
    heuristic_actor_steps: list[np.ndarray] = []
    heuristic_row_indices_for_advance: list[np.ndarray] = []
    for actor, obs_step, actor_step, heuristic_rows in zip(
        actors,
        obs_steps,
        actor_steps,
        heuristic_rows_by_actor,
        strict=True,
    ):
        if heuristic_rows.size == 0:
            continue
        heuristic_actors.append(actor)
        heuristic_obs_steps.append(obs_step)
        heuristic_actor_steps.append(actor_step)
        heuristic_row_indices_for_advance.append(heuristic_rows)
    if heuristic_actors and should_track_heuristic_actor_hidden_state():
        central_advance_actor_rows(
            actors=heuristic_actors,
            obs_steps=heuristic_obs_steps,
            actor_steps=heuristic_actor_steps,
            row_indices_by_actor=heuristic_row_indices_for_advance,
        )
