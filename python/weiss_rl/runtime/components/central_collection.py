"""Central actor-unroll collection for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.central_actor_step import execute_central_actor_step
from weiss_rl.runtime.components.central_collection_setup import (
    actors_have_single_layout,
    build_central_actor_collection_setup,
)
from weiss_rl.runtime.components.central_finalization import finalize_central_actor_unrolls
from weiss_rl.runtime.components.central_policy_phase import run_central_policy_phase
from weiss_rl.runtime.components.collection.central_actor_step_context import (
    CentralActorStepCallbacks,
    CentralActorStepInputs,
    CentralActorStepPolicyInputs,
    CentralActorStepRuntimeContext,
)
from weiss_rl.runtime.components.collection.central_step_inputs import prepare_central_step_inputs
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeCentralCollectionMixin:
    def _collect_actor_unrolls_central(self: Any, actors: Sequence[_ActorState]) -> list[RuntimeUnroll]:
        central_started = time.perf_counter()
        if not actors:
            return []
        if not actors_have_single_layout(actors):
            return [self._collect_actor_unroll(actor) for actor in actors]

        setup = build_central_actor_collection_setup(
            actors=actors,
            config=self.config,
            observation_dim=int(self.observation_dim),
            trajectory_retention_enabled=bool(getattr(self, "_trajectory_retention_enabled", False)),
            actor_inference_model=_actor_inference_model,
        )
        callbacks = CentralActorStepCallbacks(
            policy_train_mask_for_actor=self._policy_train_mask_for_actor,
            trajectory_retention_mask_for_actor=self._trajectory_retention_mask_for_actor,
            ensure_legal_action_meta=self._ensure_legal_action_meta,
            teacher_labels_from_ids=self._teacher_labels_from_ids,
            teacher_labels_from_mask=self._teacher_labels_from_mask,
            update_outcomes=self._update_outcomes,
            assign_episode_roles=self._assign_episode_roles,
            reset_done_rows=self._reset_done_rows,
        )
        batches = setup.batches
        for step_index in range(setup.time_steps):
            step_inputs = prepare_central_step_inputs(
                actors=actors,
                batches=batches,
                states_by_actor=setup.states_by_actor,
                step_index=step_index,
                batch_size=setup.batch_size,
                filter_action_surface_for_batch=self._filter_action_surface_for_batch,
            )
            batches = step_inputs.batches
            obs_storage_steps = step_inputs.obs_storage_steps
            obs_steps = step_inputs.obs_steps
            actor_steps = step_inputs.actor_steps
            policy_outputs = run_central_policy_phase(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                states_by_actor=setup.states_by_actor,
                batch_size=setup.batch_size,
                action_dim=int(self.action_dim),
                structured_central_packed=setup.structured_central_packed,
                disable_mirror_policy_fusion=bool(getattr(self, "_disable_mirror_policy_fusion", False)),
                opponent_heuristic_policy_ids=tuple(getattr(self, "_opponent_heuristic_policies", {}).keys()),
                record_batch_timer_ms=self._record_batch_timer_ms,
                central_sample_policy_rows_ids=self._central_sample_policy_rows_ids,
                central_advance_actor_rows=self._central_advance_actor_rows,
                should_track_heuristic_actor_hidden_state=self._should_track_heuristic_actor_hidden_state,
                apply_opponent_rows_ids=self._apply_opponent_rows_ids,
                ensure_legal_action_meta=self._ensure_legal_action_meta,
                central_forward_all_rows=self._central_forward_all_rows,
                overwrite_central_outputs_with_configured_opponents=self._overwrite_central_outputs_with_configured_opponents,
            )

            next_batches: list[DecisionBoundaryBatch] = []
            for actor_index, (actor, batch, obs_storage_step, actor_step, logits_step, value_step) in enumerate(
                zip(
                    actors,
                    batches,
                    obs_storage_steps,
                    actor_steps,
                    policy_outputs.logits_steps,
                    policy_outputs.value_steps,
                    strict=True,
                )
            ):
                state = setup.states_by_actor[int(actor.actor_id)]
                next_batches.append(
                    execute_central_actor_step(
                        actor=actor,
                        batch=batch,
                        state=state,
                        inputs=CentralActorStepInputs(
                            step_index=step_index,
                            obs_storage_step=obs_storage_step,
                            actor_step=actor_step,
                            policy=CentralActorStepPolicyInputs(
                                actor_index=actor_index,
                                logits_step=logits_step,
                                value_step=value_step,
                                structured_central_packed=setup.structured_central_packed,
                                structured_action_steps=policy_outputs.action_steps,
                                structured_logp_steps=policy_outputs.logp_steps,
                            ),
                        ),
                        runtime=CentralActorStepRuntimeContext(
                            config=self.config,
                            action_family_index=getattr(self, "_action_family_index", None),
                            device=self._device,
                            timeout_limits=setup.timeout_limits_by_actor[int(actor.actor_id)],
                            callbacks=callbacks,
                        ),
                    )
                )
            batches = next_batches

        return finalize_central_actor_unrolls(
            actors=actors,
            batches=batches,
            states_by_actor=setup.states_by_actor,
            batch_size=setup.batch_size,
            action_dim=int(self.action_dim),
            structured_central_packed=setup.structured_central_packed,
            values_required=bool(getattr(self, "_actor_behavior_values_required", True)),
            central_started=central_started,
            central_value_actor_rows=self._central_value_actor_rows,
            central_forward_all_rows=self._central_forward_all_rows,
            overwrite_central_outputs_with_configured_opponents=self._overwrite_central_outputs_with_configured_opponents,
        )
