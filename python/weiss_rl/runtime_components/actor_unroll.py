"""Generic single-actor unroll collection for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.core.masking import logits_for_sampling_temperature, masked_logp_from_mask
from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)
from weiss_rl.runtime_components.batching import (
    optional_legal_action_meta as _optional_legal_action_meta,
)
from weiss_rl.runtime_components.batching import (
    require_ids_offsets as _require_ids_offsets,
)
from weiss_rl.runtime_components.batching import (
    require_mask as _require_mask,
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
        teacher_family = np.full((T, N), -1, dtype=np.int32)
        teacher_slot = np.full((T, N), -1, dtype=np.int32)
        teacher_move_source = np.full((T, N), -1, dtype=np.int32)
        teacher_attack_type = np.full((T, N), -1, dtype=np.int32)
        teacher_action = np.full((T, N), -1, dtype=np.int32)
        teacher_valid = np.zeros((T, N), dtype=np.bool_)
        trajectory_retention_valid = (
            np.zeros((T, N), dtype=np.bool_) if bool(getattr(self, "_trajectory_retention_enabled", False)) else None
        )
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets: list[np.ndarray] = [np.array([0], dtype=np.uint32)]
        mask_steps: list[np.ndarray] = []
        counters = _collector_counter_template()
        action_sequence_state = make_action_sequence_state(N)
        timeout_limits = _timeout_limits_for_env(actor.env)

        batch = actor.current_batch
        initial_hidden_state = actor.seat_hidden.detach().cpu().numpy().copy()
        for step_index in range(T):
            batch = self._filter_action_surface_for_batch(
                batch,
                counters=counters,
                action_sequence_state=action_sequence_state,
            )
            obs_storage_step = np.array(batch.obs, copy=True)
            obs_step = np.array(batch.obs, dtype=np.float32, copy=True)
            actor_step = np.array(batch.actor, dtype=np.int64, copy=True)
            if obs_step.shape != (N, self.observation_dim):
                raise RuntimeError(f"unexpected actor obs shape: {obs_step.shape}")
            if np.any((actor_step != 0) & (actor_step != 1)):
                raise RuntimeError(f"actor runtime only supports live seat rows, got {actor_step.tolist()}")
            opponent_context_index[step_index] = opponent_context_indices_for_model(
                actor.model,
                actor.opponent_policy_id_by_env,
                batch_size=N,
            )
            focal_rows = actor_step == actor.focal_seat_by_env
            _accumulate_actor_role_row_counters(
                counters=counters,
                actor_step=actor_step,
                focal_seat_by_env=actor.focal_seat_by_env,
            )
            value_step = np.zeros((N,), dtype=np.float32)
            action_step = np.zeros((N,), dtype=np.int64)
            logp_step = np.zeros((N,), dtype=np.float32)
            reward_legal_ids: np.ndarray | None = None
            reward_legal_offsets: np.ndarray | None = None
            reward_legal_meta: np.ndarray | None = None
            reward_legal_mask: np.ndarray | None = None
            policy_train_mask[step_index] = self._policy_train_mask_for_actor(
                actor=actor,
                focal_rows=focal_rows,
            )
            logits_step = np.empty((N, self.action_dim), dtype=np.float32)

            if actor.layout_name == "i16_legal_ids":
                legal_ids, legal_offsets = _require_ids_offsets(batch)
                legal_action_meta = self._ensure_legal_action_meta(legal_ids, _optional_legal_action_meta(batch))
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
                    decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                    obs_step=obs_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                counters["packed_candidate_count"] += int(np.asarray(legal_ids).shape[0])
                packed_legal_ids = np.array(legal_ids, dtype=np.int64, copy=True)
                packed_legal_offsets = np.array(legal_offsets, dtype=np.int64, copy=True)
                reward_legal_ids = packed_legal_ids
                reward_legal_offsets = packed_legal_offsets
                reward_legal_meta = (
                    None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
                )
                offset_base = int(packed_offsets[-1][-1])
                if self._use_simulator_fused_logits_step and hasattr(actor.env, "step_sample_from_logits_with_logp"):
                    packed_ids.append(np.array(legal_ids, dtype=np.uint32, copy=True))
                    if legal_action_meta is not None:
                        packed_meta.append(np.array(legal_action_meta, dtype=np.uint16, copy=True))
                    packed_offsets.append(np.array(legal_offsets[1:] + offset_base, dtype=np.uint32, copy=True))
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_ids(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        logits_out=logits_step,
                        values_out=value_step,
                        actions_out=None,
                        logp_out=None,
                        rng=actor.rng,
                        sample_actions=False,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                    sampling_logits_step = logits_for_sampling_temperature(
                        logits_step,
                        temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    )
                    env_started = time.perf_counter()
                    next_batch, fused_actions, fused_logp = actor.env.step_sample_from_logits_with_logp(
                        sampling_logits_step,
                        sample_seeds,
                    )
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    action_step = np.asarray(fused_actions, dtype=np.int64)
                    logp_step = np.asarray(fused_logp, dtype=np.float32)
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)
                else:
                    packed_ids.append(np.array(legal_ids, dtype=np.uint32, copy=True))
                    if legal_action_meta is not None:
                        packed_meta.append(np.array(legal_action_meta, dtype=np.uint16, copy=True))
                    packed_offsets.append(np.array(legal_offsets[1:] + offset_base, dtype=np.uint32, copy=True))
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_ids(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        logits_out=None,
                        values_out=value_step,
                        actions_out=action_step,
                        logp_out=logp_step,
                        rng=actor.rng,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    env_started = time.perf_counter()
                    self._maybe_debug_validate_env_step_packed_actions(
                        actor=actor,
                        source_label="collect:packed",
                        actions=action_step,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                    )
                    next_batch = actor.env.step(action_step.astype(np.uint32, copy=False))
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)
            else:
                legal_mask = _require_mask(batch)
                teacher_started = time.perf_counter()
                (
                    teacher_family_step,
                    teacher_slot_step,
                    teacher_move_source_step,
                    teacher_attack_type_step,
                    teacher_action_step,
                    teacher_valid_step,
                ) = self._teacher_labels_from_mask(
                    focal_rows=focal_rows,
                    decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                    obs_step=obs_step,
                    legal_mask=np.asarray(legal_mask, dtype=np.bool_),
                    counters=counters,
                )
                counters["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                if self._use_simulator_fused_logits_step:
                    current_legal_mask = np.asarray(legal_mask, dtype=np.bool_).copy()
                    reward_legal_mask = current_legal_mask
                    mask_steps.append(current_legal_mask)
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_mask(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_mask=current_legal_mask,
                        logits_out=logits_step,
                        values_out=value_step,
                        actions_out=None,
                        logp_out=None,
                        rng=actor.rng,
                        sample_actions=False,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                    sampling_logits_step = logits_for_sampling_temperature(
                        logits_step,
                        temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    )
                    env_started = time.perf_counter()
                    next_batch, fused_actions = actor.env.step_sample_from_logits(sampling_logits_step, sample_seeds)
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    action_step = np.asarray(fused_actions, dtype=np.int64)
                    logp_step = masked_logp_from_mask(
                        sampling_logits_step,
                        current_legal_mask,
                        action_step.astype(np.uint32, copy=False),
                        pass_action_id=self.config.pass_action_id,
                    )
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_mask=current_legal_mask,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)
                else:
                    legal_mask_array = np.array(legal_mask, dtype=np.bool_, copy=True)
                    reward_legal_mask = legal_mask_array
                    mask_steps.append(legal_mask_array)
                    policy_started = time.perf_counter()
                    self._fill_policy_outputs_mask(
                        actor=actor,
                        obs_step=obs_step,
                        actor_step=actor_step,
                        focal_rows=focal_rows,
                        legal_mask=legal_mask,
                        logits_out=None,
                        values_out=value_step,
                        actions_out=action_step,
                        logp_out=logp_step,
                        rng=actor.rng,
                    )
                    counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
                    env_started = time.perf_counter()
                    next_batch = actor.env.step(action_step.astype(np.uint32, copy=False))
                    counters["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=counters,
                        state=action_sequence_state,
                        actions=action_step,
                        legal_mask=legal_mask_array,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    counters["actor_action_summary_ms"] += int((time.perf_counter() - summary_started) * 1000.0)
            done = np.logical_or(next_batch.terminated, next_batch.truncated)
            if reward_legal_mask is not None:
                reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                    np.asarray(next_batch.reward, dtype=np.float32),
                    np.asarray(action_step, dtype=np.int64),
                    pass_action_id=self.config.pass_action_id,
                    penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                    legal_mask=reward_legal_mask,
                )
            else:
                reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                    np.asarray(next_batch.reward, dtype=np.float32),
                    np.asarray(action_step, dtype=np.int64),
                    pass_action_id=self.config.pass_action_id,
                    penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                    legal_ids=reward_legal_ids,
                    legal_offsets=reward_legal_offsets,
                    legal_action_meta=reward_legal_meta,
                    ignored_alternative_family_ids=_pass_penalty_ignored_family_ids(
                        getattr(self, "_action_family_index", None)
                    ),
                )
            counters["pass_with_nonpass_penalty_count"] += penalty_count
            counters["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
            if reward_legal_ids is not None and reward_legal_offsets is not None:
                family_index = getattr(self, "_action_family_index", {})
                reward_step, penalty_count, penalty_total_micros = _apply_mulligan_penalty(
                    reward_step,
                    np.asarray(action_step, dtype=np.int64),
                    penalty=float(getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)),
                    legal_ids=reward_legal_ids,
                    legal_offsets=reward_legal_offsets,
                    legal_action_meta=reward_legal_meta,
                    mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                    mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
                )
                counters["mulligan_select_with_confirm_penalty_count"] += penalty_count
                counters["mulligan_select_with_confirm_penalty_total_micros"] += penalty_total_micros

            obs[step_index] = obs_storage_step
            actions[step_index] = action_step.astype(np.uint16, copy=False)
            rewards[step_index] = reward_step
            terminated[step_index] = np.asarray(next_batch.terminated, dtype=np.bool_)
            truncated[step_index] = np.asarray(next_batch.truncated, dtype=np.bool_)
            to_play_seat[step_index] = actor_step.astype(np.int8, copy=False)
            behavior_logp[step_index] = logp_step
            values[step_index] = value_step
            episode_seed[step_index] = np.asarray(next_batch.episode_seed, dtype=np.uint64)
            teacher_family[step_index] = teacher_family_step
            teacher_slot[step_index] = teacher_slot_step
            teacher_move_source[step_index] = teacher_move_source_step
            teacher_attack_type[step_index] = teacher_attack_type_step
            teacher_action[step_index] = teacher_action_step
            teacher_valid[step_index] = teacher_valid_step
            retention_valid_step = self._trajectory_retention_mask_for_actor(actor=actor, focal_rows=focal_rows)
            if trajectory_retention_valid is not None and retention_valid_step is not None:
                trajectory_retention_valid[step_index] = retention_valid_step
                counters["trajectory_retention_rows"] += int(np.count_nonzero(retention_valid_step))

            if np.any(done):
                _accumulate_timeout_counters(
                    counters=counters,
                    batch=next_batch,
                    done=done,
                    timeout_limits=timeout_limits,
                )
                self._update_outcomes(
                    actor=actor,
                    acting_seat=actor_step,
                    terminal_batch=next_batch,
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
                batch = self._reset_done_rows(actor, done.astype(np.bool_, copy=False))
                counters["actor_done_reset_ms"] += int((time.perf_counter() - reset_started) * 1000.0)
            else:
                batch = next_batch

        actor.current_batch = batch
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
            legal_actions=(
                LegalActionBatch.from_packed(
                    np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                    np.concatenate(packed_offsets, axis=0),
                    meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
                    action_space=int(self.action_dim),
                )
                if actor.layout_name == "i16_legal_ids"
                else LegalActionBatch.from_mask(np.stack(mask_steps, axis=0), action_space=int(self.action_dim))
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
            trajectory_retention_valid=trajectory_retention_valid,
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
            + teacher_family.nbytes
            + teacher_slot.nbytes
            + teacher_move_source.nbytes
            + teacher_attack_type.nbytes
            + teacher_action.nbytes
            + teacher_valid.nbytes
            + (0 if trajectory_retention_valid is None else trajectory_retention_valid.nbytes)
            + bootstrap_obs.nbytes
            + bootstrap_actor.nbytes
            + bootstrap_value.nbytes
        )
        _merge_simulator_timing_counters(counters, actor.env)
        counters["collect_actor_unroll_ms"] += int((time.perf_counter() - unroll_started) * 1000.0)
        actor.next_unroll_seq += 1
        return unroll
