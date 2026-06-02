"""One-actor step execution for central batched collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.central_actor_action import (
    execute_mask_central_actor_action,
    execute_packed_central_actor_action,
)
from weiss_rl.runtime.components.collection.central_actor_action_context import (
    CentralActorActionInputs,
    MaskCentralActorActionCallbacks,
    PackedCentralActorActionCallbacks,
    PackedCentralActorActionMode,
)
from weiss_rl.runtime.components.collection.central_actor_step_context import (
    CentralActorStepInputs,
    CentralActorStepRuntimeContext,
)
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.collector_unroll_storage import store_collector_step
from weiss_rl.runtime.components.terminal_episode_reset import reset_terminal_episode_rows

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def execute_central_actor_step(
    *,
    actor: _ActorState,
    batch: Any,
    state: CollectorUnrollState,
    inputs: CentralActorStepInputs,
    runtime: CentralActorStepRuntimeContext,
) -> Any:
    callbacks = runtime.callbacks
    policy = inputs.policy
    obs_step = np.asarray(inputs.obs_storage_step, dtype=np.float32)
    focal_rows = inputs.actor_step == actor.focal_seat_by_env
    state.policy_train_mask[inputs.step_index] = callbacks.policy_train_mask_for_actor(
        actor=actor, focal_rows=focal_rows
    )
    action_inputs = CentralActorActionInputs(
        actor=actor,
        batch=batch,
        state=state,
        obs_step=obs_step,
        focal_rows=focal_rows,
        logits_step=policy.logits_step,
        config=runtime.config,
        action_family_index=runtime.action_family_index,
    )
    if actor.layout_name == "i16_legal_ids":
        action_result = execute_packed_central_actor_action(
            inputs=action_inputs,
            mode=PackedCentralActorActionMode(
                actor_index=policy.actor_index,
                structured_central_packed=policy.structured_central_packed,
                structured_action_steps=policy.structured_action_steps,
                structured_logp_steps=policy.structured_logp_steps,
            ),
            callbacks=PackedCentralActorActionCallbacks(
                ensure_legal_action_meta=callbacks.ensure_legal_action_meta,
                teacher_labels_from_ids=callbacks.teacher_labels_from_ids,
            ),
        )
    else:
        action_result = execute_mask_central_actor_action(
            inputs=action_inputs,
            callbacks=MaskCentralActorActionCallbacks(
                teacher_labels_from_mask=callbacks.teacher_labels_from_mask,
            ),
        )

    next_batch = action_result.next_batch
    done = np.logical_or(next_batch.terminated, next_batch.truncated)
    retention_valid = callbacks.trajectory_retention_mask_for_actor(actor=actor, focal_rows=focal_rows)
    store_collector_step(
        step_index=inputs.step_index,
        obs_storage=state.obs,
        actions_storage=state.actions,
        rewards_storage=state.rewards,
        terminated_storage=state.terminated,
        truncated_storage=state.truncated,
        to_play_seat_storage=state.to_play_seat,
        behavior_logp_storage=state.behavior_logp,
        values_storage=state.values,
        episode_seed_storage=state.episode_seed,
        teacher_family_storage=state.teacher_family,
        teacher_slot_storage=state.teacher_slot,
        teacher_move_source_storage=state.teacher_move_source,
        teacher_attack_type_storage=state.teacher_attack_type,
        teacher_action_storage=state.teacher_action,
        teacher_valid_storage=state.teacher_valid,
        trajectory_retention_storage=state.trajectory_retention_valid,
        obs_step=inputs.obs_storage_step,
        actions=action_result.actions,
        rewards=action_result.rewards,
        terminated=next_batch.terminated,
        truncated=next_batch.truncated,
        actor_step=inputs.actor_step,
        behavior_logp=action_result.behavior_logp,
        values=policy.value_step,
        episode_seed=next_batch.episode_seed,
        teacher_labels=action_result.teacher_labels,
        retention_valid=retention_valid,
        counters=state.counters,
    )

    if not np.any(done):
        return next_batch

    return reset_terminal_episode_rows(
        actor=actor,
        next_batch=next_batch,
        acting_seat=inputs.actor_step,
        done=done,
        counters=state.counters,
        timeout_limits=runtime.timeout_limits,
        action_sequence_state=state.action_sequence_state,
        device=runtime.device,
        update_outcomes=callbacks.update_outcomes,
        assign_episode_roles=callbacks.assign_episode_roles,
        reset_done_rows=callbacks.reset_done_rows,
    )
