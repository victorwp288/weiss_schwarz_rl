"""Central actor-unroll collection for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.core.masking import (
    logits_for_sampling_temperature,
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
)
from weiss_rl.diagnostics.action_diagnostics import (
    make_action_sequence_state,
    reset_action_sequence_state,
    update_action_summary_from_ids,
    update_action_summary_from_mask,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
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
from weiss_rl.runtime_components.policy_ids import MIRROR_OPPONENT_POLICY_ID as _MIRROR_OPPONENT_POLICY_ID
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


class QueueRuntimeCentralCollectionMixin:
    def _collect_actor_unrolls_central(self: Any, actors: Sequence[_ActorState]) -> list[RuntimeUnroll]:
        central_started = time.perf_counter()
        if not actors:
            return []
        if len({str(actor.layout_name) for actor in actors}) != 1:
            return [self._collect_actor_unroll(actor) for actor in actors]

        T = int(self.config.unroll_length)
        N = int(self.config.envs_per_actor)
        obs_dtype = np.asarray(actors[0].current_batch.obs).dtype
        state_by_actor: dict[int, dict[str, Any]] = {}
        for actor in actors:
            state_by_actor[int(actor.actor_id)] = {
                "obs": np.zeros((T, N, self.observation_dim), dtype=obs_dtype),
                "actions": np.zeros((T, N), dtype=np.uint16),
                "rewards": np.zeros((T, N), dtype=np.float32),
                "terminated": np.zeros((T, N), dtype=np.bool_),
                "truncated": np.zeros((T, N), dtype=np.bool_),
                "to_play_seat": np.zeros((T, N), dtype=np.int8),
                "behavior_logp": np.zeros((T, N), dtype=np.float32),
                "values": np.zeros((T, N), dtype=np.float32),
                "episode_seed": np.zeros((T, N), dtype=np.uint64),
                "policy_train_mask": np.zeros((T, N), dtype=np.bool_),
                "opponent_context_index": np.zeros((T, N), dtype=np.int16),
                "teacher_family": np.full((T, N), -1, dtype=np.int32),
                "teacher_slot": np.full((T, N), -1, dtype=np.int32),
                "teacher_move_source": np.full((T, N), -1, dtype=np.int32),
                "teacher_attack_type": np.full((T, N), -1, dtype=np.int32),
                "teacher_action": np.full((T, N), -1, dtype=np.int32),
                "teacher_valid": np.zeros((T, N), dtype=np.bool_),
                "trajectory_retention_valid": (
                    np.zeros((T, N), dtype=np.bool_)
                    if bool(getattr(self, "_trajectory_retention_enabled", False))
                    else None
                ),
                "packed_ids": [],
                "packed_meta": [],
                "packed_offsets": [np.array([0], dtype=np.uint32)],
                "mask_steps": [],
                "initial_hidden_state": actor.seat_hidden.detach().cpu().numpy().copy(),
                "counters": _collector_counter_template(),
                "action_sequence_state": make_action_sequence_state(N),
            }
        timeout_limits_by_actor = {int(actor.actor_id): _timeout_limits_for_env(actor.env) for actor in actors}

        batches = [actor.current_batch for actor in actors]
        structured_central_packed = bool(
            all(actor.layout_name == "i16_legal_ids" for actor in actors)
            and bool(getattr(_actor_inference_model(actors[0]), "supports_legal_candidate_scoring", False))
        )
        for step_index in range(T):
            batches = [
                self._filter_action_surface_for_batch(
                    batch,
                    counters=state_by_actor[int(actor.actor_id)]["counters"],
                    action_sequence_state=state_by_actor[int(actor.actor_id)]["action_sequence_state"],
                )
                for actor, batch in zip(actors, batches, strict=True)
            ]
            obs_storage_steps = [np.array(batch.obs, copy=True) for batch in batches]
            obs_steps = [np.array(batch.obs, dtype=np.float32, copy=True) for batch in batches]
            actor_steps = [np.array(batch.actor, dtype=np.int64, copy=True) for batch in batches]
            for actor in actors:
                state_by_actor[int(actor.actor_id)]["opponent_context_index"][step_index] = (
                    opponent_context_indices_for_model(
                        actor.model,
                        actor.opponent_policy_id_by_env,
                        batch_size=N,
                    )
                )
            if structured_central_packed:
                action_steps = [np.zeros((N,), dtype=np.int64) for _ in actors]
                logp_steps = [np.zeros((N,), dtype=np.float32) for _ in actors]
                value_steps = [np.zeros((N,), dtype=np.float32) for _ in actors]
                policy_row_indices = [
                    np.flatnonzero(actor_step == actor.focal_seat_by_env)
                    for actor, actor_step in zip(actors, actor_steps, strict=True)
                ]
                for actor, row_indices in zip(actors, policy_row_indices, strict=True):
                    state_by_actor[int(actor.actor_id)]["counters"]["focal_row_count"] += int(row_indices.shape[0])
                fuse_mirror_policy_rows = not bool(getattr(self, "_disable_mirror_policy_fusion", False))
                heuristic_policy_ids = tuple(getattr(self, "_opponent_heuristic_policies", {}).keys())
                heuristic_rows_by_actor: list[np.ndarray] = []
                mirror_rows_by_actor: list[np.ndarray] = []
                residual_rows_by_actor: list[np.ndarray] = []
                for actor, actor_step in zip(actors, actor_steps, strict=True):
                    opponent_indices = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    if opponent_indices.size == 0:
                        heuristic_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        mirror_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        residual_rows_by_actor.append(np.zeros((0,), dtype=np.int64))
                        continue
                    opponent_policy_ids = np.asarray(
                        actor.opponent_policy_id_by_env[opponent_indices],
                        dtype=object,
                    )
                    heuristic_mask = (
                        np.isin(opponent_policy_ids, heuristic_policy_ids)
                        if heuristic_policy_ids
                        else np.zeros(opponent_policy_ids.shape, dtype=np.bool_)
                    )
                    mirror_mask = opponent_policy_ids == _MIRROR_OPPONENT_POLICY_ID
                    heuristic_rows_by_actor.append(opponent_indices[heuristic_mask])
                    mirror_rows_by_actor.append(opponent_indices[mirror_mask])
                    residual_rows_by_actor.append(opponent_indices[~(heuristic_mask | mirror_mask)])
                for actor, heuristic_rows, mirror_rows, residual_rows in zip(
                    actors,
                    heuristic_rows_by_actor,
                    mirror_rows_by_actor,
                    residual_rows_by_actor,
                    strict=True,
                ):
                    state_by_actor[int(actor.actor_id)]["counters"]["opponent_row_count"] += int(
                        heuristic_rows.shape[0] + mirror_rows.shape[0] + residual_rows.shape[0]
                    )
                sampled_policy_rows_by_actor = [
                    (
                        np.concatenate((focal_rows, mirror_rows), axis=0).astype(np.int64, copy=False)
                        if fuse_mirror_policy_rows and mirror_rows.size > 0
                        else focal_rows
                    )
                    for focal_rows, mirror_rows in zip(policy_row_indices, mirror_rows_by_actor, strict=True)
                ]
                forward_started = time.perf_counter()
                if any(rows.size > 0 for rows in sampled_policy_rows_by_actor):
                    self._central_sample_policy_rows_ids(
                        actors=actors,
                        batches=batches,
                        obs_steps=obs_steps,
                        actor_steps=actor_steps,
                        row_indices_by_actor=sampled_policy_rows_by_actor,
                        values_outs=value_steps,
                        actions_outs=action_steps,
                        logp_outs=logp_steps,
                    )
                self._record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
                per_actor_forward_ms = int(((time.perf_counter() - forward_started) * 1000.0) / max(len(actors), 1))
                for state in state_by_actor.values():
                    state["counters"]["actor_policy_forward_ms"] += per_actor_forward_ms

                overwrite_started = time.perf_counter()
                if fuse_mirror_policy_rows and heuristic_policy_ids:
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
                    if heuristic_actors and self._should_track_heuristic_actor_hidden_state():
                        self._central_advance_actor_rows(
                            actors=heuristic_actors,
                            obs_steps=heuristic_obs_steps,
                            actor_steps=heuristic_actor_steps,
                            row_indices_by_actor=heuristic_row_indices_for_advance,
                        )
                for (
                    actor,
                    batch,
                    obs_step,
                    actor_step,
                    value_step,
                    action_step,
                    logp_step,
                    heuristic_rows,
                    residual_rows,
                ) in zip(
                    actors,
                    batches,
                    obs_steps,
                    actor_steps,
                    value_steps,
                    action_steps,
                    logp_steps,
                    heuristic_rows_by_actor,
                    residual_rows_by_actor,
                    strict=True,
                ):
                    legal_ids, legal_offsets = _require_ids_offsets(batch)
                    legal_action_meta = self._ensure_legal_action_meta(legal_ids, _optional_legal_action_meta(batch))
                    if fuse_mirror_policy_rows:
                        if heuristic_rows.size > 0:
                            self._apply_opponent_rows_ids(
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
                        opponent_rows = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    if opponent_rows.size > 0:
                        self._apply_opponent_rows_ids(
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
                self._record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
                per_actor_overwrite_ms = int(((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1))
                for state in state_by_actor.values():
                    state["counters"]["fixed_opponent_routing_ms"] += per_actor_overwrite_ms
                logits_steps: list[np.ndarray | None] = [None for _ in actors]
            else:
                logits_steps = [np.empty((N, self.action_dim), dtype=np.float32) for _ in actors]
                value_steps = [np.empty((N,), dtype=np.float32) for _ in actors]
                for actor, actor_step in zip(actors, actor_steps, strict=True):
                    focal_rows = np.flatnonzero(actor_step == actor.focal_seat_by_env)
                    opponent_rows = np.flatnonzero(actor_step != actor.focal_seat_by_env)
                    state_by_actor[int(actor.actor_id)]["counters"]["focal_row_count"] += int(focal_rows.shape[0])
                    state_by_actor[int(actor.actor_id)]["counters"]["opponent_row_count"] += int(opponent_rows.shape[0])
                forward_started = time.perf_counter()
                self._central_forward_all_rows(
                    actors=actors,
                    batches=batches,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    logits_outs=cast(Sequence[np.ndarray], logits_steps),
                    values_outs=value_steps,
                )
                self._record_batch_timer_ms("central_focal_policy", time.perf_counter() - forward_started)
                per_actor_forward_ms = int(((time.perf_counter() - forward_started) * 1000.0) / max(len(actors), 1))
                for state in state_by_actor.values():
                    state["counters"]["actor_policy_forward_ms"] += per_actor_forward_ms

                overwrite_started = time.perf_counter()
                self._overwrite_central_outputs_with_configured_opponents(
                    actors=actors,
                    batches=batches,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    logits_outs=cast(Sequence[np.ndarray | None], logits_steps),
                    values_outs=value_steps,
                )
                per_actor_overwrite_ms = int(((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1))
                for state in state_by_actor.values():
                    state["counters"]["fixed_opponent_routing_ms"] += per_actor_overwrite_ms

            next_batches: list[DecisionBoundaryBatch] = []
            for actor_index, (actor, batch, obs_storage_step, actor_step, logits_step, value_step) in enumerate(
                zip(
                    actors,
                    batches,
                    obs_storage_steps,
                    actor_steps,
                    logits_steps,
                    value_steps,
                    strict=True,
                )
            ):
                state = state_by_actor[int(actor.actor_id)]
                obs_step = np.asarray(obs_storage_step, dtype=np.float32)
                focal_rows = actor_step == actor.focal_seat_by_env
                state["policy_train_mask"][step_index] = self._policy_train_mask_for_actor(
                    actor=actor,
                    focal_rows=focal_rows,
                )
                if actor.layout_name == "i16_legal_ids":
                    legal_ids, legal_offsets = _require_ids_offsets(batch)
                    legal_action_meta = self._ensure_legal_action_meta(legal_ids, _optional_legal_action_meta(batch))
                    teacher_started = time.perf_counter()
                    (
                        teacher_family,
                        teacher_slot,
                        teacher_move_source,
                        teacher_attack_type,
                        teacher_action,
                        teacher_valid,
                    ) = self._teacher_labels_from_ids(
                        focal_rows=focal_rows,
                        decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                        obs_step=obs_step,
                        legal_ids=legal_ids,
                        legal_offsets=legal_offsets,
                        legal_action_meta=legal_action_meta,
                        counters=state["counters"],
                    )
                    state["counters"]["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                    state["counters"]["packed_candidate_count"] += int(np.asarray(legal_ids).shape[0])
                    packed_legal_ids = np.array(legal_ids, dtype=np.int64, copy=True)
                    packed_legal_offsets = np.array(legal_offsets, dtype=np.int64, copy=True)
                    offset_base = int(state["packed_offsets"][-1][-1])
                    state["packed_ids"].append(np.array(legal_ids, dtype=np.uint32, copy=True))
                    if legal_action_meta is not None:
                        state["packed_meta"].append(np.array(legal_action_meta, dtype=np.uint16, copy=True))
                    state["packed_offsets"].append(
                        np.array(legal_offsets[1:] + offset_base, dtype=np.uint32, copy=True)
                    )
                    if structured_central_packed:
                        action_step = np.asarray(action_steps[actor_index], dtype=np.int64)
                        logp_step = np.asarray(logp_steps[actor_index], dtype=np.float32)
                        env_started = time.perf_counter()
                        next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                        state["counters"]["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    elif hasattr(actor.env, "step_sample_from_logits_with_logp"):
                        sample_seeds = actor.rng.integers(0, np.iinfo(np.int64).max, size=N, dtype=np.int64)
                        sampling_logits_step = logits_for_sampling_temperature(
                            cast(np.ndarray, logits_step),
                            temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                        )
                        env_started = time.perf_counter()
                        next_batch, fused_actions, fused_logp = actor.env.step_sample_from_logits_with_logp(
                            sampling_logits_step,
                            sample_seeds,
                        )
                        state["counters"]["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                        action_step = np.asarray(fused_actions, dtype=np.int64)
                        logp_step = np.asarray(fused_logp, dtype=np.float32)
                    else:
                        env_started = time.perf_counter()
                        action_step, logp_step, _entropy = sample_actions_from_legal_ids(
                            cast(np.ndarray, logits_step),
                            legal_ids,
                            legal_offsets,
                            rng=actor.rng,
                            pass_action_id=self.config.pass_action_id,
                            temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                        )
                        next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                        state["counters"]["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_ids(
                        counters=state["counters"],
                        state=state["action_sequence_state"],
                        actions=action_step,
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    state["counters"]["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                else:
                    legal_mask = _require_mask(batch)
                    legal_mask_array = np.array(legal_mask, dtype=np.bool_, copy=True)
                    teacher_started = time.perf_counter()
                    (
                        teacher_family,
                        teacher_slot,
                        teacher_move_source,
                        teacher_attack_type,
                        teacher_action,
                        teacher_valid,
                    ) = self._teacher_labels_from_mask(
                        focal_rows=focal_rows,
                        decision_kind=np.asarray(batch.decision_kind, dtype=np.int32),
                        obs_step=obs_step,
                        legal_mask=legal_mask_array,
                        counters=state["counters"],
                    )
                    state["counters"]["teacher_label_ms"] += int((time.perf_counter() - teacher_started) * 1000.0)
                    state["mask_steps"].append(legal_mask_array)
                    env_started = time.perf_counter()
                    action_step, logp_step, _entropy = sample_actions_from_mask(
                        logits_step,
                        legal_mask,
                        rng=actor.rng,
                        pass_action_id=self.config.pass_action_id,
                        temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    )
                    state["counters"]["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    env_started = time.perf_counter()
                    next_batch = actor.env.step(np.asarray(action_step, dtype=np.uint32))
                    state["counters"]["actor_env_step_ms"] += int((time.perf_counter() - env_started) * 1000.0)
                    summary_started = time.perf_counter()
                    update_action_summary_from_mask(
                        counters=state["counters"],
                        state=state["action_sequence_state"],
                        actions=action_step,
                        legal_mask=legal_mask_array,
                        pass_action_id=self.config.pass_action_id,
                        main_move_action=getattr(next_batch, "main_move_action", None),
                    )
                    state["counters"]["actor_action_summary_ms"] += int(
                        (time.perf_counter() - summary_started) * 1000.0
                    )
                done = np.logical_or(next_batch.terminated, next_batch.truncated)
                if actor.layout_name == "i16_legal_ids":
                    reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                        np.asarray(next_batch.reward, dtype=np.float32),
                        np.asarray(action_step, dtype=np.int64),
                        pass_action_id=self.config.pass_action_id,
                        penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        legal_action_meta=legal_action_meta,
                        ignored_alternative_family_ids=_pass_penalty_ignored_family_ids(
                            getattr(self, "_action_family_index", None)
                        ),
                    )
                else:
                    reward_step, penalty_count, penalty_total_micros = _apply_pass_penalty(
                        np.asarray(next_batch.reward, dtype=np.float32),
                        np.asarray(action_step, dtype=np.int64),
                        pass_action_id=self.config.pass_action_id,
                        penalty=float(getattr(self.config, "pass_with_nonpass_penalty", 0.0)),
                        legal_mask=legal_mask_array,
                    )
                state["counters"]["pass_with_nonpass_penalty_count"] += penalty_count
                state["counters"]["pass_with_nonpass_penalty_total_micros"] += penalty_total_micros
                if actor.layout_name == "i16_legal_ids":
                    family_index = getattr(self, "_action_family_index", {})
                    reward_step, penalty_count, penalty_total_micros = _apply_mulligan_penalty(
                        reward_step,
                        np.asarray(action_step, dtype=np.int64),
                        penalty=float(getattr(self.config, "mulligan_select_with_confirm_penalty", 0.0)),
                        legal_ids=packed_legal_ids,
                        legal_offsets=packed_legal_offsets,
                        legal_action_meta=legal_action_meta,
                        mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                        mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
                    )
                    state["counters"]["mulligan_select_with_confirm_penalty_count"] += penalty_count
                    state["counters"]["mulligan_select_with_confirm_penalty_total_micros"] += penalty_total_micros

                state["obs"][step_index] = obs_storage_step
                state["actions"][step_index] = np.asarray(action_step, dtype=np.uint16)
                state["rewards"][step_index] = reward_step
                state["terminated"][step_index] = np.asarray(next_batch.terminated, dtype=np.bool_)
                state["truncated"][step_index] = np.asarray(next_batch.truncated, dtype=np.bool_)
                state["to_play_seat"][step_index] = actor_step.astype(np.int8, copy=False)
                state["behavior_logp"][step_index] = np.asarray(logp_step, dtype=np.float32)
                state["values"][step_index] = value_step
                state["episode_seed"][step_index] = np.asarray(next_batch.episode_seed, dtype=np.uint64)
                state["teacher_family"][step_index] = teacher_family
                state["teacher_slot"][step_index] = teacher_slot
                state["teacher_move_source"][step_index] = teacher_move_source
                state["teacher_attack_type"][step_index] = teacher_attack_type
                state["teacher_action"][step_index] = teacher_action
                state["teacher_valid"][step_index] = teacher_valid
                retention_valid = self._trajectory_retention_mask_for_actor(actor=actor, focal_rows=focal_rows)
                if state["trajectory_retention_valid"] is not None and retention_valid is not None:
                    state["trajectory_retention_valid"][step_index] = retention_valid
                    state["counters"]["trajectory_retention_rows"] += int(np.count_nonzero(retention_valid))

                if np.any(done):
                    _accumulate_timeout_counters(
                        counters=state["counters"],
                        batch=next_batch,
                        done=done,
                        timeout_limits=timeout_limits_by_actor[int(actor.actor_id)],
                    )
                    self._update_outcomes(
                        actor=actor,
                        acting_seat=actor_step,
                        terminal_batch=next_batch,
                        done=done.astype(np.bool_, copy=False),
                        counters=state["counters"],
                    )
                    reset_started = time.perf_counter()
                    done_mask = torch.as_tensor(done, dtype=torch.bool, device=self._device)
                    self._assign_episode_roles(actor, done.astype(np.bool_, copy=False), counters=state["counters"])
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
                    reset_action_sequence_state(state["action_sequence_state"], done.astype(np.bool_, copy=False))
                    next_batch = self._reset_done_rows(actor, done.astype(np.bool_, copy=False))
                    state["counters"]["actor_done_reset_ms"] += int((time.perf_counter() - reset_started) * 1000.0)
                next_batches.append(next_batch)
            batches = next_batches

        bootstrap_obs_steps = [np.asarray(batch.obs, dtype=np.float32) for batch in batches]
        bootstrap_actor_steps = [np.asarray(batch.actor, dtype=np.int64) for batch in batches]
        bootstrap_values = [np.zeros((N,), dtype=np.float32) for _ in actors]
        if bool(getattr(self, "_actor_behavior_values_required", True)):
            bootstrap_started = time.perf_counter()
            if structured_central_packed:
                self._central_value_actor_rows(
                    actors=actors,
                    obs_steps=bootstrap_obs_steps,
                    actor_steps=bootstrap_actor_steps,
                    row_indices_by_actor=[np.arange(N, dtype=np.int64) for _ in actors],
                    values_outs=bootstrap_values,
                )
            else:
                self._central_forward_all_rows(
                    actors=actors,
                    batches=batches,
                    obs_steps=bootstrap_obs_steps,
                    actor_steps=bootstrap_actor_steps,
                    logits_outs=[np.empty((N, self.action_dim), dtype=np.float32) for _ in actors],
                    values_outs=bootstrap_values,
                )
            bootstrap_forward_ms = int(((time.perf_counter() - bootstrap_started) * 1000.0) / max(len(actors), 1))
            for state in state_by_actor.values():
                state["counters"]["actor_bootstrap_ms"] += bootstrap_forward_ms

            if not structured_central_packed:
                overwrite_started = time.perf_counter()
                self._overwrite_central_outputs_with_configured_opponents(
                    actors=actors,
                    batches=batches,
                    obs_steps=bootstrap_obs_steps,
                    actor_steps=bootstrap_actor_steps,
                    logits_outs=[None for _ in actors],
                    values_outs=bootstrap_values,
                )
                bootstrap_overwrite_ms = int(((time.perf_counter() - overwrite_started) * 1000.0) / max(len(actors), 1))
                for state in state_by_actor.values():
                    state["counters"]["fixed_opponent_routing_ms"] += bootstrap_overwrite_ms

        unrolls: list[RuntimeUnroll] = []
        for actor, batch, bootstrap_value in zip(actors, batches, bootstrap_values, strict=True):
            state = state_by_actor[int(actor.actor_id)]
            actor.current_batch = batch
            state["counters"]["copied_bytes_estimate"] += int(
                state["obs"].nbytes
                + state["actions"].nbytes
                + state["rewards"].nbytes
                + state["terminated"].nbytes
                + state["truncated"].nbytes
                + state["to_play_seat"].nbytes
                + state["behavior_logp"].nbytes
                + state["values"].nbytes
                + state["episode_seed"].nbytes
                + state["policy_train_mask"].nbytes
                + state["opponent_context_index"].nbytes
                + state["teacher_family"].nbytes
                + state["teacher_slot"].nbytes
                + state["teacher_move_source"].nbytes
                + state["teacher_attack_type"].nbytes
                + state["teacher_action"].nbytes
                + state["teacher_valid"].nbytes
                + (0 if state["trajectory_retention_valid"] is None else state["trajectory_retention_valid"].nbytes)
                + np.asarray(batch.obs, dtype=np.float32).nbytes
                + np.asarray(batch.actor, dtype=np.int64).nbytes
                + np.asarray(bootstrap_value, dtype=np.float32).nbytes
            )
            _merge_simulator_timing_counters(state["counters"], actor.env)
            state["counters"]["collect_actor_unroll_ms"] += int(
                ((time.perf_counter() - central_started) * 1000.0) / max(len(actors), 1)
            )
            unrolls.append(
                RuntimeUnroll(
                    actor_id=actor.actor_id,
                    unroll_seq=actor.next_unroll_seq,
                    behavior_policy_version=actor.snapshot_version,
                    unroll_hash=_hash_unroll(
                        actions=state["actions"],
                        rewards=state["rewards"],
                        episode_seed=state["episode_seed"],
                    ),
                    obs=state["obs"],
                    actions=state["actions"],
                    rewards=state["rewards"],
                    terminated=state["terminated"],
                    truncated=state["truncated"],
                    to_play_seat=state["to_play_seat"],
                    behavior_logp=state["behavior_logp"],
                    values=state["values"],
                    legal_actions=(
                        LegalActionBatch.from_packed(
                            np.concatenate(state["packed_ids"], axis=0)
                            if state["packed_ids"]
                            else np.zeros((0,), dtype=np.uint32),
                            np.concatenate(state["packed_offsets"], axis=0),
                            meta=(np.concatenate(state["packed_meta"], axis=0) if state["packed_meta"] else None),
                            action_space=int(self.action_dim),
                        )
                        if actor.layout_name == "i16_legal_ids"
                        else LegalActionBatch.from_mask(
                            np.stack(state["mask_steps"], axis=0),
                            action_space=int(self.action_dim),
                        )
                    ),
                    bootstrap_obs=np.asarray(batch.obs, dtype=np.float32),
                    bootstrap_actor=np.asarray(batch.actor, dtype=np.int64),
                    bootstrap_value=np.asarray(bootstrap_value, dtype=np.float32),
                    initial_hidden_state=state["initial_hidden_state"],
                    final_hidden_state=actor.seat_hidden.detach().cpu().numpy().copy(),
                    episode_seed=state["episode_seed"],
                    policy_train_mask=state["policy_train_mask"],
                    opponent_context_index=state["opponent_context_index"],
                    teacher_family=state["teacher_family"],
                    teacher_slot=state["teacher_slot"],
                    teacher_move_source=state["teacher_move_source"],
                    teacher_attack_type=state["teacher_attack_type"],
                    teacher_action=state["teacher_action"],
                    teacher_valid=state["teacher_valid"],
                    trajectory_retention_valid=state["trajectory_retention_valid"],
                    behavior_logits=None,
                    counters=dict(state["counters"]),
                )
            )
            actor.next_unroll_seq += 1
        return unrolls
