"""All-heuristic actor rollout fast paths for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
)
from weiss_rl.envs.decision_env import _pack_batch
from weiss_rl.runtime_components.batching import (
    optional_legal_action_meta as _optional_legal_action_meta,
)
from weiss_rl.runtime_components.batching import (
    require_ids_offsets as _require_ids_offsets,
)
from weiss_rl.runtime_components.counters import (
    accumulate_actor_role_row_counters as _accumulate_actor_role_row_counters,
)
from weiss_rl.runtime_components.counters import (
    accumulate_timeout_counters as _accumulate_timeout_counters,
)
from weiss_rl.runtime_components.counters import (
    collector_counter_template as _collector_counter_template,
)
from weiss_rl.runtime_components.counters import (
    merge_simulator_timing_counters as _merge_simulator_timing_counters,
)
from weiss_rl.runtime_components.counters import (
    timeout_limits_for_env as _timeout_limits_for_env,
)
from weiss_rl.runtime_components.hashing import hash_unroll as _hash_unroll
from weiss_rl.runtime_components.opponent_context import (
    initial_seat_hidden_for_opponents,
    opponent_context_indices_for_model,
)
from weiss_rl.runtime_components.reward_shaping import (
    apply_mulligan_select_with_confirm_penalty as _apply_mulligan_penalty,
)
from weiss_rl.runtime_components.reward_shaping import apply_pass_with_nonpass_penalty as _apply_pass_penalty
from weiss_rl.runtime_components.reward_shaping import (
    pass_penalty_ignored_alternative_family_ids as _pass_penalty_ignored_family_ids,
)
from weiss_rl.runtime_components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime_components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeHeuristicRolloutMixin:
    def _collect_actor_unroll_all_heuristic_ids_native_rollout(self: Any, actor: _ActorState) -> RuntimeUnroll:
        unroll_started = time.perf_counter()
        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        counters = _collector_counter_template()
        action_sequence_state = make_action_sequence_state(N)
        initial_hidden_state = actor.seat_hidden.detach().cpu().numpy().copy()
        teacher_labels_enabled = self._teacher_guidance_active_for_collection()

        pool = getattr(actor.env, "pool", None)
        if pool is None:
            raise RuntimeError("heuristic native rollout requires a pooled simulator env")
        rollout_into = getattr(pool, "rollout_heuristic_public_into_i16_legal_ids", None)
        reset_done_into = getattr(pool, "reset_done_into_i16_legal_ids", None)
        if not callable(rollout_into) or not callable(reset_done_into):
            raise RuntimeError(
                "heuristic native rollout requires "
                "pool.rollout_heuristic_public_into_i16_legal_ids(...) and "
                "pool.reset_done_into_i16_legal_ids(...)"
            )

        weiss_sim = __import__("weiss_sim")
        trajectory = weiss_sim.BatchOutTrajectoryI16LegalIds(T, N)
        rollout_started = time.perf_counter()
        rollout_into(T, trajectory)
        rollout_elapsed = time.perf_counter() - rollout_started
        actor.env._record_python_timing(
            "python_native_heuristic_rollout",
            int(rollout_elapsed * 1_000_000_000.0),
        )
        counters["actor_env_step_ms"] += int(rollout_elapsed * 1000.0)

        obs = np.asarray(trajectory.obs, dtype=np.float32)
        actions = np.asarray(trajectory.actions, dtype=np.uint16)
        rewards = np.asarray(trajectory.rewards, dtype=np.float32).copy()
        terminated = np.asarray(trajectory.terminated, dtype=np.bool_)
        truncated = np.asarray(trajectory.truncated, dtype=np.bool_)
        to_play_seat = np.asarray(trajectory.actor, dtype=np.int8)
        behavior_logp = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        episode_seed_src = getattr(trajectory, "episode_seed", None)
        if episode_seed_src is None:
            episode_seed = np.asarray(trajectory.spec_hash, dtype=np.uint64)
        else:
            episode_seed = np.asarray(episode_seed_src, dtype=np.uint64)
        policy_train_mask = np.zeros((T, N), dtype=np.bool_)
        opponent_context_index = np.zeros((T, N), dtype=np.int16)
        teacher_family = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_slot = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_move_source = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_attack_type = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_action = np.full((T, N), -1, dtype=np.int32) if teacher_labels_enabled else None
        teacher_valid = np.zeros((T, N), dtype=np.bool_) if teacher_labels_enabled else None
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        legal_ids_all = trajectory.legal_ids
        legal_offsets_all = trajectory.legal_offsets
        legal_meta_all = getattr(trajectory, "legal_action_meta", None)
        decision_kind_all = trajectory.decision_kind
        main_move_action_all = getattr(trajectory, "main_move_action", None)

        for step_index in range(T):
            current_actor = np.asarray(to_play_seat[step_index], dtype=np.int64)
            if np.any((current_actor != 0) & (current_actor != 1)):
                raise RuntimeError(f"actor runtime only supports live seat rows, got {current_actor.tolist()}")
            opponent_context_index[step_index] = opponent_context_indices_for_model(
                getattr(actor, "model", getattr(self, "model", None)),
                getattr(actor, "opponent_policy_id_by_env", [None] * N),
                batch_size=N,
            )
            focal_rows = current_actor == actor.focal_seat_by_env
            _accumulate_actor_role_row_counters(
                counters=counters,
                actor_step=current_actor,
                focal_seat_by_env=actor.focal_seat_by_env,
            )
            policy_train_mask[step_index] = self._policy_train_mask_for_actor(
                actor=actor,
                focal_rows=focal_rows,
                include_mirror_opponent_rows=False,
            )

            step_offsets = np.asarray(legal_offsets_all[step_index], dtype=np.uint32)
            used = 0 if step_offsets.size == 0 else int(step_offsets[-1])
            step_ids = np.asarray(legal_ids_all[step_index], dtype=np.uint32)[:used]
            step_meta = (
                None if legal_meta_all is None else np.asarray(legal_meta_all[step_index], dtype=np.uint16)[:used]
            )

            teacher_family_step = teacher_slot_step = teacher_move_source_step = teacher_attack_type_step = (
                teacher_action_step
            ) = teacher_valid_step = None
            if teacher_labels_enabled:
                teacher_started = time.perf_counter()
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_move_source_step,
                    teacher_attack_type_step,
                    teacher_action_step,
                    teacher_valid_step,
                ) = self._teacher_labels_from_ids(
                    focal_rows=focal_rows,
                    decision_kind=np.asarray(decision_kind_all[step_index], dtype=np.int32),
                    obs_step=np.asarray(obs[step_index], dtype=np.float32),
                    legal_ids=step_ids,
                    legal_offsets=step_offsets,
                    legal_action_meta=step_meta,
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
            counters["packed_candidate_count"] += int(step_ids.shape[0])

            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(np.array(step_ids, dtype=np.uint32, copy=True))
            if step_meta is not None:
                packed_meta.append(np.array(step_meta, dtype=np.uint16, copy=True))
            packed_offsets.append(np.asarray(step_offsets[1:] + offset_base, dtype=np.uint32))

            if teacher_labels_enabled:
                assert teacher_family is not None and teacher_slot is not None
                assert (
                    teacher_move_source is not None
                    and teacher_attack_type is not None
                    and teacher_action is not None
                    and teacher_valid is not None
                )
                teacher_family[step_index] = teacher_family_step
                teacher_slot[step_index] = teacher_slot_step
                teacher_move_source[step_index] = teacher_move_source_step
                teacher_attack_type[step_index] = teacher_attack_type_step
                teacher_action[step_index] = teacher_action_step
                teacher_valid[step_index] = teacher_valid_step

            summary_started = time.perf_counter()
            update_action_summary_from_ids(
                counters=counters,
                state=action_sequence_state,
                actions=np.asarray(actions[step_index], dtype=np.int64),
                legal_ids=np.asarray(step_ids, dtype=np.int64),
                legal_offsets=np.asarray(step_offsets, dtype=np.int64),
                pass_action_id=self.config.pass_action_id,
                main_move_action=(
                    None if main_move_action_all is None else np.asarray(main_move_action_all[step_index])
                ),
            )
            counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)

            done = np.logical_or(terminated[step_index], truncated[step_index])
            if np.any(done):
                self._update_outcomes_from_transition_arrays(
                    actor=actor,
                    acting_seat=current_actor,
                    rewards=np.asarray(trajectory.rewards[step_index], dtype=np.float32),
                    truncated=np.asarray(truncated[step_index], dtype=np.bool_),
                    done=done,
                )
                self._assign_episode_roles(actor, done.astype(np.bool_, copy=False), counters=counters)
                reset_action_sequence_state(action_sequence_state, done.astype(np.bool_, copy=False))
            reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                np.asarray(trajectory.rewards[step_index], dtype=np.float32),
                np.asarray(actions[step_index], dtype=np.int64),
                pass_action_id=self.config.pass_action_id,
                penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                legal_ids=np.asarray(step_ids, dtype=np.int64),
                legal_offsets=np.asarray(step_offsets, dtype=np.int64),
                legal_action_meta=step_meta,
                ignored_alternative_family_ids=_pass_penalty_ignored_family_ids(
                    getattr(self, "_action_family_index", None)
                ),
            )
            rewards[step_index] = reward_step
            counters["pass_with_nonpass_penalty_count"] += penalty_count
            counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
            family_index = getattr(self, "_action_family_index", {})
            rewards[step_index], penalty_count, penalty_total_micros = _apply_mulligan_penalty(
                rewards[step_index],
                np.asarray(actions[step_index], dtype=np.int64),
                penalty=float(getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)),
                legal_ids=np.asarray(step_ids, dtype=np.int64),
                legal_offsets=np.asarray(step_offsets, dtype=np.int64),
                legal_action_meta=step_meta,
                mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
            )
            counters["mulligan_select_with_confirm_penalty_count"] += penalty_count
            counters["mulligan_select_with_confirm_penalty_total_micros"] += penalty_total_micros

        step_out = getattr(actor.env, "_step_out", None)
        if step_out is None:
            step_out = actor.env._require_step_out(weiss_sim)
        snapshot_started = time.perf_counter()
        reset_done_into(np.zeros((N,), dtype=np.bool_), step_out)
        snapshot_elapsed = time.perf_counter() - snapshot_started
        actor.env._record_python_timing("python_reset_done", int(snapshot_elapsed * 1_000_000_000.0))
        actor.env._handle_engine_status(step_out, weiss_sim=None)
        batch = self._sync_actor_batch_from_step_out(
            actor=actor,
            step_out=step_out,
            pool=pool,
        )
        bootstrap_obs = np.asarray(batch.obs, dtype=np.float32)
        bootstrap_actor = np.asarray(batch.actor, dtype=np.int64)
        bootstrap_value = np.zeros((batch.obs.shape[0],), dtype=np.float32)

        unroll = RuntimeUnroll(
            actor_id=actor.actor_id,
            unroll_seq=actor.next_unroll_seq,
            behavior_policy_version=actor.snapshot_version,
            unroll_hash=_hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed),
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            values=values,
            legal_actions=LegalActionBatch.from_packed(
                np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                np.concatenate(packed_offsets, axis=0),
                meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
                action_space=int(self.action_dim),
            ),
            bootstrap_obs=bootstrap_obs,
            bootstrap_actor=bootstrap_actor,
            bootstrap_value=bootstrap_value,
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
            behavior_logits=None,
            counters=counters,
        )
        counters["copied_bytes_estimate"] += int(
            obs.nbytes
            + actions.nbytes
            + rewards.nbytes
            + terminated.nbytes
            + truncated.nbytes
            + to_play_seat.nbytes
            + behavior_logp.nbytes
            + values.nbytes
            + episode_seed.nbytes
            + policy_train_mask.nbytes
            + opponent_context_index.nbytes
            + bootstrap_obs.nbytes
            + bootstrap_actor.nbytes
            + bootstrap_value.nbytes
        )
        if teacher_family is not None:
            assert teacher_slot is not None
            assert teacher_move_source is not None
            assert teacher_attack_type is not None
            assert teacher_action is not None
            assert teacher_valid is not None
            counters["copied_bytes_estimate"] += int(
                teacher_family.nbytes
                + teacher_slot.nbytes
                + teacher_move_source.nbytes
                + teacher_attack_type.nbytes
                + teacher_action.nbytes
                + teacher_valid.nbytes
            )
        _merge_simulator_timing_counters(counters, actor.env)
        counters["collect_actor_unroll_ms"] += int((time.perf_counter() - unroll_started) * 1000.0)
        actor.next_unroll_seq += 1
        return unroll

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

            teacher_family_step = teacher_slot_step = teacher_move_source_step = teacher_attack_type_step = (
                teacher_action_step
            ) = teacher_valid_step = None
            if teacher_labels_enabled:
                teacher_started = time.perf_counter()
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_move_source_step,
                    teacher_attack_type_step,
                    teacher_action_step,
                    teacher_valid_step,
                ) = self._teacher_labels_from_ids(
                    focal_rows=focal_rows,
                    decision_kind=current_decision_kind,
                    obs_step=current_obs,
                    legal_ids=current_legal_ids,
                    legal_offsets=current_legal_offsets,
                    legal_action_meta=current_legal_action_meta,
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
            counters["packed_candidate_count"] += int(current_legal_ids.shape[0])

            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(np.array(current_legal_ids, dtype=np.uint32, copy=True))
            if current_legal_action_meta is not None:
                packed_meta.append(np.array(current_legal_action_meta, dtype=np.uint16, copy=True))
            packed_offsets.append(np.array(current_legal_offsets[1:] + offset_base, dtype=np.uint32, copy=True))

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
            reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                step_rewards,
                action_step,
                pass_action_id=self.config.pass_action_id,
                penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                legal_ids=np.asarray(current_legal_ids, dtype=np.int64),
                legal_offsets=np.asarray(current_legal_offsets, dtype=np.int64),
                legal_action_meta=current_legal_action_meta,
                ignored_alternative_family_ids=_pass_penalty_ignored_family_ids(
                    getattr(self, "_action_family_index", None)
                ),
            )
            counters["pass_with_nonpass_penalty_count"] += penalty_count
            counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
            family_index = getattr(self, "_action_family_index", {})
            reward_step, penalty_count, penalty_total_micros = _apply_mulligan_penalty(
                reward_step,
                action_step,
                penalty=float(getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)),
                legal_ids=np.asarray(current_legal_ids, dtype=np.int64),
                legal_offsets=np.asarray(current_legal_offsets, dtype=np.int64),
                legal_action_meta=current_legal_action_meta,
                mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
            )
            counters["mulligan_select_with_confirm_penalty_count"] += penalty_count
            counters["mulligan_select_with_confirm_penalty_total_micros"] += penalty_total_micros
            step_terminated = np.asarray(step_out.terminated, dtype=np.bool_)
            step_truncated = np.asarray(step_out.truncated, dtype=np.bool_)
            step_episode_seed = np.asarray(pool.episode_seed_batch(), dtype=np.uint64)
            done = np.logical_or(step_terminated, step_truncated)

            obs[step_index] = current_obs_storage
            actions[step_index] = action_step.astype(np.uint16, copy=False)
            rewards[step_index] = reward_step
            terminated[step_index] = step_terminated
            truncated[step_index] = step_truncated
            to_play_seat[step_index] = current_actor.astype(np.int8, copy=False)
            behavior_logp[step_index] = logp_step
            values[step_index] = value_step
            episode_seed[step_index] = step_episode_seed
            if teacher_labels_enabled:
                assert teacher_family is not None and teacher_slot is not None
                assert (
                    teacher_move_source is not None
                    and teacher_attack_type is not None
                    and teacher_action is not None
                    and teacher_valid is not None
                )
                teacher_family[step_index] = teacher_family_step
                teacher_slot[step_index] = teacher_slot_step
                teacher_move_source[step_index] = teacher_move_source_step
                teacher_attack_type[step_index] = teacher_attack_type_step
                teacher_action[step_index] = teacher_action_step
                teacher_valid[step_index] = teacher_valid_step

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
                done_mask = torch.as_tensor(done, dtype=torch.bool, device=self._device)
                self._assign_episode_roles(actor, done.astype(np.bool_, copy=False), counters=counters)
                reset_hidden = initial_seat_hidden_for_opponents(
                    actor.model,
                    int(np.count_nonzero(done)),
                    device=self._device,
                    opponent_policy_ids=actor.opponent_policy_id_by_env[done],
                )
                actor.seat_hidden[done_mask] = reset_hidden
                actor.opponent_hidden[done_mask] = initial_seat_hidden_for_opponents(
                    actor.model,
                    int(np.count_nonzero(done)),
                    device=self._device,
                )
                reset_action_sequence_state(action_sequence_state, done.astype(np.bool_, copy=False))
                reset_done_into(np.ascontiguousarray(done, dtype=np.bool_), step_out)
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
        bootstrap_value = np.zeros((batch.obs.shape[0],), dtype=np.float32)
        bootstrap_obs = np.asarray(batch.obs, dtype=np.float32)
        bootstrap_actor = np.asarray(batch.actor, dtype=np.int64)
        valid_bootstrap_rows = (bootstrap_actor == 0) | (bootstrap_actor == 1)
        if bool(getattr(self, "_actor_behavior_values_required", True)) and np.any(valid_bootstrap_rows):
            bootstrap_started = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.amp.autocast(
                    device_type=self._device.type,
                    enabled=self._actor_amp_enabled,
                ),
            ):
                actor_model = _actor_inference_model(actor)
                value_seat_aware = getattr(actor_model, "value_seat_aware", None)
                if callable(value_seat_aware):
                    bootstrap_value_tensor = value_seat_aware(
                        torch.as_tensor(bootstrap_obs[valid_bootstrap_rows], device=self._device),
                        torch.as_tensor(bootstrap_actor[valid_bootstrap_rows], device=self._device, dtype=torch.long),
                        actor.seat_hidden[valid_bootstrap_rows],
                    )
                else:
                    _, bootstrap_value_tensor, _ = actor_model.forward_seat_aware(
                        torch.as_tensor(bootstrap_obs[valid_bootstrap_rows], device=self._device),
                        torch.as_tensor(bootstrap_actor[valid_bootstrap_rows], device=self._device, dtype=torch.long),
                        actor.seat_hidden[valid_bootstrap_rows],
                    )
            bootstrap_value[valid_bootstrap_rows] = (
                bootstrap_value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            counters["actor_bootstrap_ms"] += int((time.perf_counter() - bootstrap_started) * 1000.0)

        unroll = RuntimeUnroll(
            actor_id=actor.actor_id,
            unroll_seq=actor.next_unroll_seq,
            behavior_policy_version=actor.snapshot_version,
            unroll_hash=_hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed),
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            to_play_seat=to_play_seat,
            behavior_logp=behavior_logp,
            values=values,
            legal_actions=LegalActionBatch.from_packed(
                np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                np.concatenate(packed_offsets, axis=0),
                meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
                action_space=int(self.action_dim),
            ),
            bootstrap_obs=bootstrap_obs,
            bootstrap_actor=bootstrap_actor,
            bootstrap_value=bootstrap_value,
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
            behavior_logits=None,
            counters=counters,
        )
        counters["copied_bytes_estimate"] += int(
            obs.nbytes
            + actions.nbytes
            + rewards.nbytes
            + terminated.nbytes
            + truncated.nbytes
            + to_play_seat.nbytes
            + behavior_logp.nbytes
            + values.nbytes
            + episode_seed.nbytes
            + policy_train_mask.nbytes
            + opponent_context_index.nbytes
            + bootstrap_obs.nbytes
            + bootstrap_actor.nbytes
            + bootstrap_value.nbytes
        )
        if teacher_family is not None:
            assert teacher_slot is not None
            assert teacher_move_source is not None
            assert teacher_attack_type is not None
            assert teacher_action is not None
            assert teacher_valid is not None
            counters["copied_bytes_estimate"] += int(
                teacher_family.nbytes
                + teacher_slot.nbytes
                + teacher_move_source.nbytes
                + teacher_attack_type.nbytes
                + teacher_action.nbytes
                + teacher_valid.nbytes
            )
        _merge_simulator_timing_counters(counters, actor.env)
        counters["collect_actor_unroll_ms"] += int((time.perf_counter() - unroll_started) * 1000.0)
        actor.next_unroll_seq += 1
        return unroll
