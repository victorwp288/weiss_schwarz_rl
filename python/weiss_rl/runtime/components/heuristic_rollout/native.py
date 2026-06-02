"""Native simulator all-heuristic actor rollout implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
)
from weiss_rl.runtime.components.bootstrap import bootstrap_fields_from_batch
from weiss_rl.runtime.components.collector_step_legal import capture_packed_array_step_legal
from weiss_rl.runtime.components.collector_unroll_storage import build_collector_runtime_unroll
from weiss_rl.runtime.components.counters import (
    accumulate_actor_role_row_counters as _accumulate_actor_role_row_counters,
)
from weiss_rl.runtime.components.counters import (
    collector_counter_template as _collector_counter_template,
)
from weiss_rl.runtime.components.counters import (
    merge_simulator_timing_counters as _merge_simulator_timing_counters,
)
from weiss_rl.runtime.components.opponent_context import opponent_context_indices_for_model
from weiss_rl.runtime.components.reward_shaping import apply_collector_reward_shaping as _apply_reward_shaping
from weiss_rl.runtime.components.types import RuntimeUnroll

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeHeuristicNativeRolloutMixin:
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

            packed_legal = capture_packed_array_step_legal(
                legal_ids=step_ids,
                legal_offsets=step_offsets,
                legal_action_meta=step_meta,
                decision_kind=np.asarray(decision_kind_all[step_index], dtype=np.int32),
                focal_rows=focal_rows,
                obs_step=np.asarray(obs[step_index], dtype=np.float32),
                counters=counters,
                teacher_labels_from_ids=self._teacher_labels_from_ids if teacher_labels_enabled else None,
                packed_ids=packed_ids,
                packed_meta=packed_meta,
                packed_offsets=packed_offsets,
            )
            step_ids = packed_legal.legal_ids
            step_offsets = packed_legal.legal_offsets
            step_meta = packed_legal.legal_action_meta

            if teacher_labels_enabled:
                assert packed_legal.teacher_labels is not None
                assert teacher_family is not None and teacher_slot is not None
                assert (
                    teacher_move_source is not None
                    and teacher_attack_type is not None
                    and teacher_action is not None
                    and teacher_valid is not None
                )
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_move_source_step,
                    teacher_attack_type_step,
                    teacher_action_step,
                    teacher_valid_step,
                ) = packed_legal.teacher_labels
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
            rewards[step_index] = _apply_reward_shaping(
                np.asarray(trajectory.rewards[step_index], dtype=np.float32),
                np.asarray(actions[step_index], dtype=np.int64),
                counters=counters,
                pass_action_id=self.config.pass_action_id,
                pass_with_nonpass_penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                mulligan_select_with_confirm_penalty=float(
                    getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)
                ),
                action_family_index=getattr(self, "_action_family_index", None),
                legal_ids=np.asarray(step_ids, dtype=np.int64),
                legal_offsets=np.asarray(step_offsets, dtype=np.int64),
                legal_action_meta=step_meta,
            )

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


__all__ = ["QueueRuntimeHeuristicNativeRolloutMixin"]
