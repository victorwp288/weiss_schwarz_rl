"""Python ids-offsets all-heuristic actor rollout implementation."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
)
from weiss_rl.envs.decision_env import _pack_batch
from weiss_rl.runtime.components.bootstrap import bootstrap_fields_from_batch, collector_bootstrap_fields_for_actor
from weiss_rl.runtime.components.collector_step_legal import capture_packed_array_step_legal
from weiss_rl.runtime.components.collector_unroll_storage import build_collector_runtime_unroll, store_collector_step
from weiss_rl.runtime.components.counters import (
    accumulate_actor_role_row_counters as _accumulate_actor_role_row_counters,
)
from weiss_rl.runtime.components.counters import (
    accumulate_timeout_counters as _accumulate_timeout_counters,
)
from weiss_rl.runtime.components.counters import (
    collector_counter_template as _collector_counter_template,
)
from weiss_rl.runtime.components.counters import (
    merge_simulator_timing_counters as _merge_simulator_timing_counters,
)
from weiss_rl.runtime.components.counters import (
    timeout_limits_for_env as _timeout_limits_for_env,
)
from weiss_rl.runtime.components.done_resets import reset_actor_hidden_for_done as _reset_actor_hidden_for_done
from weiss_rl.runtime.components.legal_batching import optional_legal_action_meta as _optional_legal_action_meta
from weiss_rl.runtime.components.legal_batching import require_ids_offsets as _require_ids_offsets
from weiss_rl.runtime.components.opponent_context import opponent_context_indices_for_model
from weiss_rl.runtime.components.reward_shaping import apply_collector_reward_shaping as _apply_reward_shaping
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeHeuristicIdsFastRolloutMixin:
    def _collect_actor_unroll_all_heuristic_ids_fast(self: Any, actor: _ActorState) -> RuntimeUnroll:
        unroll_started = time.perf_counter()
        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs_dtype = np.asarray(actor.current_batch.obs).dtype
        obs = np.zeros((T, N, self.observation_dim), dtype=obs_dtype)
        actions = np.zeros((T, N), dtype=np.uint16)
        rewards = np.zeros((T, N), dtype=np.float32)
        terminated = np.zeros((T, N), dtype=np.bool_)
        truncated = np.zeros((T, N), dtype=np.bool_)
        to_play_seat = np.zeros((T, N), dtype=np.int8)
        behavior_logp = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        episode_seed = np.zeros((T, N), dtype=np.uint64)
        policy_train_mask = np.zeros((T, N), dtype=np.bool_)
        opponent_context_index = np.zeros((T, N), dtype=np.int16)
        teacher_labels_enabled = self._teacher_guidance_active_for_collection()
        teacher_family = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_slot = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_move_source = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_attack_type = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_action = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_valid = np.zeros((T, N), dtype=np.bool_) if teacher_labels_enabled else None
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        counters = _collector_counter_template()
        action_sequence_state = make_action_sequence_state(N)
        timeout_limits = _timeout_limits_for_env(actor.env)
        initial_hidden_state = actor.seat_hidden.detach().cpu().numpy().copy()

        pool = getattr(actor.env, "pool", None)
        if pool is None:
            raise RuntimeError("heuristic ids fast path requires a pooled simulator env")
        step_into = getattr(pool, "step_into_i16_legal_ids", None)
        reset_done_into = getattr(pool, "reset_done_into_i16_legal_ids", None)
        if not callable(step_into) or not callable(reset_done_into):
            raise RuntimeError(
                "heuristic ids fast path requires pool.step_into_i16_legal_ids(...) "
                "and pool.reset_done_into_i16_legal_ids(...)"
            )
        step_out = getattr(actor.env, "_step_out", None)
        if step_out is None:
            step_out = actor.env._require_step_out(__import__("weiss_sim"))

        batch = actor.current_batch

        all_rows = np.arange(N, dtype=np.int64)
        for step_index in range(T):
            batch = self._filter_action_surface_for_batch(
                batch,
                counters=counters,
                action_sequence_state=action_sequence_state,
            )
            current_obs_storage = np.array(batch.obs, copy=True)
            current_obs = np.array(batch.obs, dtype=np.float32, copy=True)
            current_actor = np.array(batch.actor, dtype=np.int64, copy=True)
            current_decision_kind = np.array(batch.decision_kind, dtype=np.int32, copy=True)
            current_legal_ids, current_legal_offsets = _require_ids_offsets(batch)
            current_legal_ids = np.array(current_legal_ids, dtype=np.uint32, copy=True)
            current_legal_offsets = np.array(current_legal_offsets, dtype=np.uint32, copy=True)
            current_legal_action_meta = self._ensure_legal_action_meta(
                current_legal_ids,
                _optional_legal_action_meta(batch),
            )
            if current_legal_action_meta is not None:
                current_legal_action_meta = np.array(current_legal_action_meta, dtype=np.uint16, copy=True)

            if current_obs.shape != (N, self.observation_dim):
                raise RuntimeError(f"unexpected actor obs shape: {current_obs.shape}")
            if np.any((current_actor != 0) & (current_actor != 1)):
                raise RuntimeError(f"actor runtime only supports live seat rows, got {current_actor.tolist()}")
            policy_ids = getattr(actor, "opponent_policy_id_by_env", None)
            if policy_ids is None:
                policy_ids = [None] * N
            opponent_context_index[step_index] = opponent_context_indices_for_model(
                getattr(actor, "model", getattr(self, "model", None)),
                policy_ids,
                batch_size=N,
            )

            focal_rows = current_actor == actor.focal_seat_by_env
            _accumulate_actor_role_row_counters(
                counters=counters,
                actor_step=current_actor,
                focal_seat_by_env=actor.focal_seat_by_env,
            )
            value_step = np.zeros((N,), dtype=np.float32)
            logp_step = np.zeros((N,), dtype=np.float32)
            policy_train_mask[step_index] = self._policy_train_mask_for_actor(
                actor=actor,
                focal_rows=focal_rows,
                include_mirror_opponent_rows=False,
            )

            packed_legal = capture_packed_array_step_legal(
                legal_ids=current_legal_ids,
                legal_offsets=current_legal_offsets,
                legal_action_meta=current_legal_action_meta,
                decision_kind=current_decision_kind,
                focal_rows=focal_rows,
                obs_step=current_obs,
                counters=counters,
                teacher_labels_from_ids=self._teacher_labels_from_ids if teacher_labels_enabled else None,
                packed_ids=packed_ids,
                packed_meta=packed_meta,
                packed_offsets=packed_offsets,
            )
            current_legal_ids = packed_legal.legal_ids
            current_legal_offsets = packed_legal.legal_offsets
            current_legal_action_meta = packed_legal.legal_action_meta

            policy_started = time.perf_counter()
            if bool(getattr(self, "_actor_behavior_values_required", True)):
                self._value_and_advance_rows(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=all_rows,
                    obs_step=current_obs,
                    actor_step=current_actor,
                    values_out=value_step,
                )
            else:
                if self._should_track_heuristic_actor_hidden_state():
                    self._advance_hidden_only(
                        model=_actor_inference_model(actor),
                        hidden_state=actor.seat_hidden,
                        row_indices=all_rows,
                        obs_step=current_obs,
                        actor_step=current_actor,
                    )
                value_step.fill(0.0)
            assert self._teacher_policy is not None
            chosen_actions = self._heuristic_public_actions_from_ids(
                actor=actor,
                heuristic_policy=self._teacher_policy,
                row_indices=all_rows,
                obs_step=current_obs,
                legal_ids=current_legal_ids,
                legal_offsets=current_legal_offsets,
                legal_action_meta=current_legal_action_meta,
                counters=counters,
            )
            self._maybe_debug_validate_sampled_packed_actions(
                source_label="process:all_heuristic",
                row_indices=all_rows,
                action_subset=np.asarray(chosen_actions, dtype=np.int64),
                legal_ids=current_legal_ids,
                legal_offsets=current_legal_offsets,
            )
            action_step = np.asarray(chosen_actions, dtype=np.int64)
            counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)

            env_started = time.perf_counter()
            step_into(np.asarray(action_step, dtype=np.uint32), step_out)
            step_elapsed = time.perf_counter() - env_started
            actor.env._record_python_timing("python_step", int(step_elapsed * 1_000_000_000.0))
            actor.env._handle_engine_status(step_out, weiss_sim=None)
            counters["actor_env_step_ms"] += int(step_elapsed * 1000.0)

            summary_started = time.perf_counter()
            update_action_summary_from_ids(
                counters=counters,
                state=action_sequence_state,
                actions=action_step,
                legal_ids=np.asarray(current_legal_ids, dtype=np.int64),
                legal_offsets=np.asarray(current_legal_offsets, dtype=np.int64),
                pass_action_id=self.config.pass_action_id,
                main_move_action=np.asarray(step_out.main_move_action),
            )
            counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)

            step_rewards = np.asarray(step_out.rewards, dtype=np.float32)
            reward_step = _apply_reward_shaping(
                step_rewards,
                action_step,
                counters=counters,
                pass_action_id=self.config.pass_action_id,
                pass_with_nonpass_penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                mulligan_select_with_confirm_penalty=float(
                    getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)
                ),
                action_family_index=getattr(self, "_action_family_index", None),
                legal_ids=np.asarray(current_legal_ids, dtype=np.int64),
                legal_offsets=np.asarray(current_legal_offsets, dtype=np.int64),
                legal_action_meta=current_legal_action_meta,
            )
            step_terminated = np.asarray(step_out.terminated, dtype=np.bool_)
            step_truncated = np.asarray(step_out.truncated, dtype=np.bool_)
            step_episode_seed = np.asarray(pool.episode_seed_batch(), dtype=np.uint64)
            done = np.logical_or(step_terminated, step_truncated)

            store_collector_step(
                step_index=step_index,
                obs_storage=obs,
                actions_storage=actions,
                rewards_storage=rewards,
                terminated_storage=terminated,
                truncated_storage=truncated,
                to_play_seat_storage=to_play_seat,
                behavior_logp_storage=behavior_logp,
                values_storage=values,
                episode_seed_storage=episode_seed,
                teacher_family_storage=teacher_family,
                teacher_slot_storage=teacher_slot,
                teacher_move_source_storage=teacher_move_source,
                teacher_attack_type_storage=teacher_attack_type,
                teacher_action_storage=teacher_action,
                teacher_valid_storage=teacher_valid,
                trajectory_retention_storage=None,
                obs_step=current_obs_storage,
                actions=action_step,
                rewards=reward_step,
                terminated=step_terminated,
                truncated=step_truncated,
                actor_step=current_actor,
                behavior_logp=logp_step,
                values=value_step,
                episode_seed=step_episode_seed,
                teacher_labels=packed_legal.teacher_labels,
                retention_valid=None,
                counters=counters,
            )

            if np.any(done):
                terminal_batch = _pack_batch(
                    step_out,
                    legality="ids_offsets",
                    pool=pool,
                    copy_arrays=True,
                )
                _accumulate_timeout_counters(
                    counters=counters,
                    batch=terminal_batch,
                    done=done,
                    timeout_limits=timeout_limits,
                )
                self._update_outcomes(
                    actor=actor,
                    acting_seat=current_actor,
                    terminal_batch=terminal_batch,
                    done=done.astype(np.bool_, copy=False),
                    counters=counters,
                )
                reset_started = time.perf_counter()
                done_array = done.astype(np.bool_, copy=False)
                self._assign_episode_roles(actor, done_array, counters=counters)
                reset_result = _reset_actor_hidden_for_done(actor=actor, done=done_array, device=self._device)
                reset_action_sequence_state(action_sequence_state, reset_result.done)
                reset_done_into(np.ascontiguousarray(reset_result.done, dtype=np.bool_), step_out)
                reset_elapsed = time.perf_counter() - reset_started
                actor.env._record_python_timing("python_reset_done", int(reset_elapsed * 1_000_000_000.0))
                actor.env._handle_engine_status(step_out, weiss_sim=None)
                counters["actor_done_reset_ms"] += int(reset_elapsed * 1000.0)

            next_batch = _pack_batch(
                step_out,
                legality="ids_offsets",
                pool=pool,
                copy_arrays=False,
            )
            if next_batch.ids_offsets is not None and next_batch.legal_action_meta is None:
                legal_meta_builder = getattr(self, "_legal_action_meta_from_ids", None)
                next_legal_action_meta = (
                    legal_meta_builder(next_batch.ids_offsets[0]) if callable(legal_meta_builder) else None
                )
                if next_legal_action_meta is not None:
                    next_batch = replace(next_batch, legal_action_meta=next_legal_action_meta)
            batch = next_batch

        batch = self._sync_actor_batch_from_step_out(
            actor=actor,
            step_out=step_out,
            pool=pool,
        )
        bootstrap_values_required = bool(getattr(self, "_actor_behavior_values_required", True))
        if bootstrap_values_required:
            bootstrap = collector_bootstrap_fields_for_actor(
                batch=batch,
                actor=actor,
                actor_model=_actor_inference_model(actor),
                bootstrap_device=self._device,
                actor_amp_enabled=self._actor_amp_enabled,
                values_required=True,
                counters=counters,
            )
        else:
            bootstrap = bootstrap_fields_from_batch(batch)

        unroll = build_collector_runtime_unroll(
            actor_id=actor.actor_id,
            unroll_seq=actor.next_unroll_seq,
            behavior_policy_version=actor.snapshot_version,
            layout_name="i16_legal_ids",
            action_dim=int(self.action_dim),
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            values=values,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
            mask_steps=[],
            bootstrap_obs=bootstrap.obs,
            bootstrap_actor=bootstrap.actor,
            bootstrap_value=bootstrap.value,
            initial_hidden_state=initial_hidden_state,
            final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
            episode_seed=episode_seed,
            policy_train_mask=policy_train_mask,
            opponent_context_index=opponent_context_index,
            teacher_family=teacher_family,
            teacher_slot=teacher_slot,
            teacher_move_source=teacher_move_source,
            teacher_attack_type=teacher_attack_type,
            teacher_action=teacher_action,
            teacher_valid=teacher_valid,
            trajectory_retention_valid=None,
            counters=counters,
            copy_counters=False,
        )
        _merge_simulator_timing_counters(counters, actor.env)
        counters["collect_actor_unroll_ms"] += int((time.perf_counter() - unroll_started) * 1000.0)
        actor.next_unroll_seq += 1
        return unroll


__all__ = ["QueueRuntimeHeuristicIdsFastRolloutMixin"]
