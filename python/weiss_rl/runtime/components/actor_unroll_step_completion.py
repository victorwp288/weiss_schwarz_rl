"""Post-policy step completion for generic actor unroll collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.runtime.components.actor_unroll_policy_execution import ActorPolicyExecutionResult
from weiss_rl.runtime.components.actor_unroll_step_inputs import ActorUnrollStepInputs
from weiss_rl.runtime.components.collector_state import CollectorUnrollState
from weiss_rl.runtime.components.collector_unroll_storage import store_collector_step
from weiss_rl.runtime.components.reward_shaping import apply_collector_reward_shaping
from weiss_rl.runtime.components.terminal_episode_reset import reset_terminal_episode_rows


@dataclass(frozen=True, slots=True)
class ActorUnrollStepCompletionInputs:
    actor: Any
    state: CollectorUnrollState
    step_index: int
    step_inputs: ActorUnrollStepInputs
    executed_policy: ActorPolicyExecutionResult
    value_step: np.ndarray
    config: Any
    action_family_index: dict[str, int] | None
    timeout_limits: Any
    device: Any


@dataclass(frozen=True, slots=True)
class ActorUnrollStepCompletionCallbacks:
    trajectory_retention_mask_for_actor: Callable[..., np.ndarray | None]
    update_outcomes: Callable[..., None]
    assign_episode_roles: Callable[..., None]
    reset_done_rows: Callable[..., Any]


def shaped_generic_actor_step_rewards(
    *,
    executed_policy: ActorPolicyExecutionResult,
    counters: dict[str, int],
    config: Any,
    action_family_index: dict[str, int] | None,
) -> np.ndarray:
    if executed_policy.reward_legal_mask is not None:
        return apply_collector_reward_shaping(
            np.asarray(executed_policy.next_batch.reward, dtype=np.float32),
            np.asarray(executed_policy.action_step, dtype=np.int64),
            counters=counters,
            pass_action_id=config.pass_action_id,
            pass_with_nonpass_penalty=float(getattr(config, "pass_with_nonpass_penalty", 0.0)),
            mulligan_select_with_confirm_penalty=float(getattr(config, "mulligan_select_with_confirm_penalty", 0.0)),
            action_family_index=action_family_index,
            legal_mask=executed_policy.reward_legal_mask,
        )
    return apply_collector_reward_shaping(
        np.asarray(executed_policy.next_batch.reward, dtype=np.float32),
        np.asarray(executed_policy.action_step, dtype=np.int64),
        counters=counters,
        pass_action_id=config.pass_action_id,
        pass_with_nonpass_penalty=float(getattr(config, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(getattr(config, "mulligan_select_with_confirm_penalty", 0.0)),
        action_family_index=action_family_index,
        legal_ids=executed_policy.reward_legal_ids,
        legal_offsets=executed_policy.reward_legal_offsets,
        legal_action_meta=executed_policy.reward_legal_meta,
    )


def complete_generic_actor_unroll_step(
    *,
    inputs: ActorUnrollStepCompletionInputs,
    callbacks: ActorUnrollStepCompletionCallbacks,
) -> Any:
    state = inputs.state
    executed = inputs.executed_policy
    next_batch = executed.next_batch
    done = np.logical_or(next_batch.terminated, next_batch.truncated)
    reward_step = shaped_generic_actor_step_rewards(
        executed_policy=executed,
        counters=state.counters,
        config=inputs.config,
        action_family_index=inputs.action_family_index,
    )

    retention_valid_step = callbacks.trajectory_retention_mask_for_actor(
        actor=inputs.actor,
        focal_rows=inputs.step_inputs.focal_rows,
    )
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
        obs_step=inputs.step_inputs.obs_storage_step,
        actions=executed.action_step,
        rewards=reward_step,
        terminated=next_batch.terminated,
        truncated=next_batch.truncated,
        actor_step=inputs.step_inputs.actor_step,
        behavior_logp=executed.logp_step,
        values=inputs.value_step,
        episode_seed=next_batch.episode_seed,
        teacher_labels=executed.teacher_labels,
        retention_valid=retention_valid_step,
        counters=state.counters,
    )

    if not np.any(done):
        return next_batch
    return reset_terminal_episode_rows(
        actor=inputs.actor,
        next_batch=next_batch,
        acting_seat=inputs.step_inputs.actor_step,
        done=done,
        counters=state.counters,
        timeout_limits=inputs.timeout_limits,
        action_sequence_state=state.action_sequence_state,
        device=inputs.device,
        update_outcomes=callbacks.update_outcomes,
        assign_episode_roles=callbacks.assign_episode_roles,
        reset_done_rows=callbacks.reset_done_rows,
    )


__all__ = [
    "ActorUnrollStepCompletionCallbacks",
    "ActorUnrollStepCompletionInputs",
    "complete_generic_actor_unroll_step",
    "shaped_generic_actor_step_rewards",
]
