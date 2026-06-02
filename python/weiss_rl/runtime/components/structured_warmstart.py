"""Structured warmstart source-mix helpers for QueueRuntime."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from weiss_rl.runtime.components.ipc_shared.ipc import serialize_state_dict_for_ipc


def _model_guidance_payload(model: Any | None) -> dict[str, float] | None:
    if model is None:
        return None
    get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
    if not callable(get_bias_scale):
        return None
    return {
        "public_heuristic_logit_bias_scale": float(get_bias_scale(scoring_mode="learner")),
        "public_heuristic_actor_logit_bias_scale": float(get_bias_scale(scoring_mode="actor")),
    }


def set_process_collector_fixed_opponents(
    runtime: Any,
    *,
    slots: np.ndarray | None,
    forced_policy_ids: Sequence[str],
    activate_teacher_heuristic: bool,
    noleague_policy_id: str,
) -> None:
    if runtime._collector_result_queue is None:
        return
    baseline_model = runtime._opponent_models.get(noleague_policy_id)
    baseline_state_dict = (
        None
        if baseline_model is None or noleague_policy_id not in forced_policy_ids
        else serialize_state_dict_for_ipc(
            {key: value.detach().cpu().clone() for key, value in baseline_model.state_dict().items()}
        )
    )
    baseline_guidance_payload = (
        None
        if baseline_model is None or noleague_policy_id not in forced_policy_ids
        else _model_guidance_payload(baseline_model)
    )
    payload = {
        "kind": "set_fixed_opponents",
        "restore_defaults": False,
        "fixed_opponent_policy_id_by_env": None if slots is None else np.asarray(slots, dtype=object).tolist(),
        "forced_policy_ids": tuple(str(policy_id) for policy_id in forced_policy_ids),
        "activate_teacher_heuristic": bool(activate_teacher_heuristic),
        "noleague_baseline_state_dict": baseline_state_dict,
        "noleague_baseline_guidance_payload": baseline_guidance_payload,
    }
    for control_queue in runtime._collector_control_queues:
        control_queue.put(payload)


def restore_process_collector_fixed_opponents(runtime: Any) -> None:
    if runtime._collector_result_queue is None:
        return
    payload = {
        "kind": "set_fixed_opponents",
        "restore_defaults": True,
    }
    for control_queue in runtime._collector_control_queues:
        control_queue.put(payload)


@contextmanager
def structured_warmstart_source_mix(
    runtime: Any,
    *,
    heuristic_policy_id: str,
    noleague_policy_id: str,
) -> Iterator[dict[str, float]]:
    inserted_teacher_heuristic = False
    if runtime._teacher_policy is not None and heuristic_policy_id not in runtime._opponent_heuristic_policies:
        runtime._opponent_heuristic_policies[heuristic_policy_id] = runtime._teacher_policy
        inserted_teacher_heuristic = True

    previous_forced_policy_ids = tuple(getattr(runtime, "_forced_fixed_opponent_policy_ids", ()))
    previous_fixed_slots = [
        (
            None
            if actor.fixed_opponent_policy_id_by_env is None
            else np.asarray(actor.fixed_opponent_policy_id_by_env, dtype=object).copy()
        )
        for actor in runtime._actors
    ]

    available_sources = ["self_play"]
    if noleague_policy_id in runtime._opponent_models:
        available_sources.append(noleague_policy_id)
    if heuristic_policy_id in runtime._opponent_heuristic_policies:
        available_sources.append(heuristic_policy_id)

    envs_per_actor = int(runtime.config.envs_per_actor)
    source_count = max(1, len(available_sources))
    counts_by_source: dict[str, int] = {}
    remaining = envs_per_actor
    for source_index, source_name in enumerate(available_sources):
        slots_left = max(1, source_count - source_index)
        count = int(np.ceil(float(remaining) / float(slots_left)))
        count = max(0, min(count, remaining))
        counts_by_source[source_name] = count
        remaining -= count

    slots = np.full((envs_per_actor,), "", dtype=object)
    cursor = 0
    for source_name in (noleague_policy_id, heuristic_policy_id):
        count = int(counts_by_source.get(source_name, 0))
        if count <= 0:
            continue
        slots[cursor : cursor + count] = source_name
        cursor += count
    forced_policy_ids = tuple(
        policy_id for policy_id in (noleague_policy_id, heuristic_policy_id) if counts_by_source.get(policy_id, 0) > 0
    )
    runtime._forced_fixed_opponent_policy_ids = forced_policy_ids
    try:
        if runtime._collector_result_queue is not None:
            set_process_collector_fixed_opponents(
                runtime,
                slots=(None if cursor <= 0 else slots.copy()),
                forced_policy_ids=forced_policy_ids,
                activate_teacher_heuristic=counts_by_source.get(heuristic_policy_id, 0) > 0,
                noleague_policy_id=noleague_policy_id,
            )
        else:
            for actor in runtime._actors:
                actor.fixed_opponent_policy_id_by_env = None if cursor <= 0 else slots.copy()
                runtime._reset_actor_state_for_fixed_opponents(actor)
        yield {
            "structured_warmstart_source_count": float(source_count),
            "structured_warmstart_self_play_envs_per_actor": float(counts_by_source.get("self_play", 0)),
            "structured_warmstart_b1_envs_per_actor": float(counts_by_source.get(noleague_policy_id, 0)),
            "structured_warmstart_b2_envs_per_actor": float(counts_by_source.get(heuristic_policy_id, 0)),
        }
    finally:
        runtime._forced_fixed_opponent_policy_ids = previous_forced_policy_ids
        if runtime._collector_result_queue is not None:
            restore_process_collector_fixed_opponents(runtime)
        else:
            for actor, saved_slots in zip(runtime._actors, previous_fixed_slots, strict=True):
                actor.fixed_opponent_policy_id_by_env = saved_slots
                runtime._reset_actor_state_for_fixed_opponents(actor)
        if inserted_teacher_heuristic:
            runtime._opponent_heuristic_policies.pop(heuristic_policy_id, None)
