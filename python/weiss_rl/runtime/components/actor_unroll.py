"""Generic single-actor unroll collection for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.actor_unroll_finalization import (
    ActorUnrollFinalizationCallbacks,
    ActorUnrollFinalizationInputs,
    finalize_generic_actor_unroll,
)
from weiss_rl.runtime.components.actor_unroll_policy_execution import (
    ActorPolicyExecutionInputs,
    MaskActorPolicyCallbacks,
    MaskActorPolicyStorage,
    PackedActorPolicyCallbacks,
    PackedActorPolicyStorage,
    execute_generic_mask_actor_policy,
    execute_generic_packed_actor_policy,
)
from weiss_rl.runtime.components.actor_unroll_step_completion import (
    ActorUnrollStepCompletionCallbacks,
    ActorUnrollStepCompletionInputs,
    complete_generic_actor_unroll_step,
)
from weiss_rl.runtime.components.actor_unroll_step_inputs import prepare_actor_unroll_step_inputs
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state
from weiss_rl.runtime.components.counters import (
    timeout_limits_for_env as _timeout_limits_for_env,
)
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeActorUnrollMixin:
    def _collect_actor_unroll(self: Any, actor: _ActorState) -> RuntimeUnroll:
        if self._can_collect_all_heuristic_ids_native_rollout(actor):
            return self._collect_actor_unroll_all_heuristic_ids_native_rollout(actor)
        if self._can_collect_all_heuristic_ids_fast(actor):
            return self._collect_actor_unroll_all_heuristic_ids_fast(actor)
        unroll_started = time.perf_counter()
        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs_dtype = np.asarray(actor.current_batch.obs).dtype
        state = allocate_collector_unroll_state(
            time_steps=T,
            batch_size=N,
            observation_dim=self.observation_dim,
            obs_dtype=obs_dtype,
            seat_hidden=actor.seat_hidden,
            trajectory_retention_enabled=bool(getattr(self, "_trajectory_retention_enabled", False)),
        )
        policy_train_mask = state.policy_train_mask
        opponent_context_index = state.opponent_context_index
        packed_ids = state.packed_ids
        packed_meta = state.packed_meta
        packed_offsets = state.packed_offsets
        mask_steps = state.mask_steps
        counters = state.counters
        action_sequence_state = state.action_sequence_state
        timeout_limits = _timeout_limits_for_env(actor.env)

        batch = actor.current_batch
        for step_index in range(T):
            step_inputs = prepare_actor_unroll_step_inputs(
                actor=actor,
                batch=batch,
                step_index=step_index,
                batch_size=N,
                observation_dim=int(self.observation_dim),
                opponent_context_index=opponent_context_index,
                counters=counters,
                action_sequence_state=action_sequence_state,
                filter_action_surface_for_batch=self._filter_action_surface_for_batch,
            )
            batch = step_inputs.batch
            obs_step = step_inputs.obs_step
            actor_step = step_inputs.actor_step
            focal_rows = step_inputs.focal_rows
            value_step = np.zeros((N,), dtype=np.float32)
            action_step = np.zeros((N,), dtype=np.int64)
            logp_step = np.zeros((N,), dtype=np.float32)
            policy_train_mask[step_index] = self._policy_train_mask_for_actor(
                actor=actor,
                focal_rows=focal_rows,
            )
            logits_step = np.empty((N, self.action_dim), dtype=np.float32)
            policy_inputs = ActorPolicyExecutionInputs(
                actor=actor,
                batch=batch,
                obs_step=obs_step,
                actor_step=actor_step,
                focal_rows=focal_rows,
                value_step=value_step,
                action_step=action_step,
                logp_step=logp_step,
                logits_step=logits_step,
                config=self.config,
                counters=counters,
                action_sequence_state=action_sequence_state,
                use_simulator_fused_logits_step=bool(self._use_simulator_fused_logits_step),
            )
            if actor.layout_name == "i16_legal_ids":
                executed_policy = execute_generic_packed_actor_policy(
                    inputs=policy_inputs,
                    callbacks=PackedActorPolicyCallbacks(
                        fill_policy_outputs_ids=self._fill_policy_outputs_ids,
                        maybe_debug_validate_env_step_packed_actions=self._maybe_debug_validate_env_step_packed_actions,
                        ensure_legal_action_meta=self._ensure_legal_action_meta,
                        teacher_labels_from_ids=self._teacher_labels_from_ids,
                    ),
                    storage=PackedActorPolicyStorage(
                        packed_ids=packed_ids,
                        packed_meta=packed_meta,
                        packed_offsets=packed_offsets,
                    ),
                )
            else:
                executed_policy = execute_generic_mask_actor_policy(
                    inputs=policy_inputs,
                    callbacks=MaskActorPolicyCallbacks(
                        fill_policy_outputs_mask=self._fill_policy_outputs_mask,
                        teacher_labels_from_mask=self._teacher_labels_from_mask,
                    ),
                    storage=MaskActorPolicyStorage(mask_steps=mask_steps),
                )
            batch = complete_generic_actor_unroll_step(
                inputs=ActorUnrollStepCompletionInputs(
                    actor=actor,
                    state=state,
                    step_index=step_index,
                    step_inputs=step_inputs,
                    executed_policy=executed_policy,
                    value_step=value_step,
                    config=self.config,
                    action_family_index=getattr(self, "_action_family_index", None),
                    timeout_limits=timeout_limits,
                    device=self._device,
                ),
                callbacks=ActorUnrollStepCompletionCallbacks(
                    trajectory_retention_mask_for_actor=self._trajectory_retention_mask_for_actor,
                    update_outcomes=self._update_outcomes,
                    assign_episode_roles=self._assign_episode_roles,
                    reset_done_rows=self._reset_done_rows,
                ),
            )

        return finalize_generic_actor_unroll(
            inputs=ActorUnrollFinalizationInputs(
                actor=actor,
                batch=batch,
                state=state,
                action_dim=int(self.action_dim),
                started_at=unroll_started,
                actor_behavior_values_required=bool(getattr(self, "_actor_behavior_values_required", True)),
                actor_amp_enabled=bool(self._actor_amp_enabled),
                bootstrap_device=self._device,
            ),
            callbacks=ActorUnrollFinalizationCallbacks(
                actor_inference_model=_actor_inference_model,
            ),
        )
