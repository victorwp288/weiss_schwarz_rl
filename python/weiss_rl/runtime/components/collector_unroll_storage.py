from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.hashing import hash_unroll
from weiss_rl.runtime.components.teacher_labels import TeacherLabelArrays
from weiss_rl.runtime.components.types import RuntimeUnroll


@dataclass(frozen=True, slots=True)
class CollectorStepStorage:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    episode_seed: np.ndarray
    teacher_family: np.ndarray | None
    teacher_slot: np.ndarray | None
    teacher_move_source: np.ndarray | None
    teacher_attack_type: np.ndarray | None
    teacher_action: np.ndarray | None
    teacher_valid: np.ndarray | None
    trajectory_retention: np.ndarray | None


@dataclass(frozen=True, slots=True)
class CollectorStepPayload:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    actor_step: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    episode_seed: np.ndarray
    teacher_labels: TeacherLabelArrays | None
    retention_valid: np.ndarray | None


@dataclass(frozen=True, slots=True)
class CollectorBootstrapArrays:
    obs: np.ndarray
    actor: np.ndarray
    value: np.ndarray


def store_collector_step(
    *,
    step_index: int,
    obs_storage: np.ndarray,
    actions_storage: np.ndarray,
    rewards_storage: np.ndarray,
    terminated_storage: np.ndarray,
    truncated_storage: np.ndarray,
    to_play_seat_storage: np.ndarray,
    behavior_logp_storage: np.ndarray,
    values_storage: np.ndarray,
    episode_seed_storage: np.ndarray,
    teacher_family_storage: np.ndarray | None,
    teacher_slot_storage: np.ndarray | None,
    teacher_move_source_storage: np.ndarray | None,
    teacher_attack_type_storage: np.ndarray | None,
    teacher_action_storage: np.ndarray | None,
    teacher_valid_storage: np.ndarray | None,
    trajectory_retention_storage: np.ndarray | None,
    obs_step: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    actor_step: np.ndarray,
    behavior_logp: np.ndarray,
    values: np.ndarray,
    episode_seed: np.ndarray,
    teacher_labels: TeacherLabelArrays | None,
    retention_valid: np.ndarray | None,
    counters: dict[str, int],
) -> None:
    write_collector_step(
        step_index=step_index,
        storage=CollectorStepStorage(
            obs=obs_storage,
            actions=actions_storage,
            rewards=rewards_storage,
            terminated=terminated_storage,
            truncated=truncated_storage,
            to_play_seat=to_play_seat_storage,
            behavior_logp=behavior_logp_storage,
            values=values_storage,
            episode_seed=episode_seed_storage,
            teacher_family=teacher_family_storage,
            teacher_slot=teacher_slot_storage,
            teacher_move_source=teacher_move_source_storage,
            teacher_attack_type=teacher_attack_type_storage,
            teacher_action=teacher_action_storage,
            teacher_valid=teacher_valid_storage,
            trajectory_retention=trajectory_retention_storage,
        ),
        payload=CollectorStepPayload(
            obs=obs_step,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            actor_step=actor_step,
            behavior_logp=behavior_logp,
            values=values,
            episode_seed=episode_seed,
            teacher_labels=teacher_labels,
            retention_valid=retention_valid,
        ),
        counters=counters,
    )


def write_collector_step(
    *,
    step_index: int,
    storage: CollectorStepStorage,
    payload: CollectorStepPayload,
    counters: dict[str, int],
) -> None:
    storage.obs[step_index] = payload.obs
    storage.actions[step_index] = np.asarray(payload.actions, dtype=np.uint16)
    storage.rewards[step_index] = np.asarray(payload.rewards, dtype=np.float32)
    storage.terminated[step_index] = np.asarray(payload.terminated, dtype=np.bool_)
    storage.truncated[step_index] = np.asarray(payload.truncated, dtype=np.bool_)
    storage.to_play_seat[step_index] = np.asarray(payload.actor_step, dtype=np.int8)
    storage.behavior_logp[step_index] = np.asarray(payload.behavior_logp, dtype=np.float32)
    storage.values[step_index] = np.asarray(payload.values, dtype=np.float32)
    storage.episode_seed[step_index] = np.asarray(payload.episode_seed, dtype=np.uint64)
    if payload.teacher_labels is not None:
        (
            teacher_family,
            teacher_slot,
            teacher_move_source,
            teacher_attack_type,
            teacher_action,
            teacher_valid,
        ) = payload.teacher_labels
        if (
            storage.teacher_family is None
            or storage.teacher_slot is None
            or storage.teacher_move_source is None
            or storage.teacher_attack_type is None
            or storage.teacher_action is None
            or storage.teacher_valid is None
        ):
            raise ValueError("teacher storage arrays are required when teacher labels are present")
        storage.teacher_family[step_index] = teacher_family
        storage.teacher_slot[step_index] = teacher_slot
        storage.teacher_move_source[step_index] = teacher_move_source
        storage.teacher_attack_type[step_index] = teacher_attack_type
        storage.teacher_action[step_index] = teacher_action
        storage.teacher_valid[step_index] = teacher_valid
    if storage.trajectory_retention is not None and payload.retention_valid is not None:
        storage.trajectory_retention[step_index] = np.asarray(payload.retention_valid, dtype=np.bool_)
        counters["trajectory_retention_rows"] += int(np.count_nonzero(payload.retention_valid))


def normalize_collector_bootstrap_arrays(
    *,
    bootstrap_obs: np.ndarray,
    bootstrap_actor: np.ndarray,
    bootstrap_value: np.ndarray,
) -> CollectorBootstrapArrays:
    return CollectorBootstrapArrays(
        obs=np.asarray(bootstrap_obs, dtype=np.float32),
        actor=np.asarray(bootstrap_actor, dtype=np.int64),
        value=np.asarray(bootstrap_value, dtype=np.float32),
    )


def legal_actions_from_collector_steps(
    *,
    layout_name: str,
    action_dim: int,
    packed_ids: list[np.ndarray],
    packed_offsets: list[np.ndarray],
    packed_meta: list[np.ndarray],
    mask_steps: list[np.ndarray],
) -> LegalActionBatch:
    if layout_name == "i16_legal_ids":
        return LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(action_dim),
        )
    return LegalActionBatch.from_mask(np.stack(mask_steps, axis=0), action_space=int(action_dim))


def estimate_collector_copied_bytes(*arrays: np.ndarray | None) -> int:
    return int(sum(0 if array is None else np.asarray(array).nbytes for array in arrays))


def build_collector_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    layout_name: str,
    action_dim: int,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    to_play_seat: np.ndarray,
    behavior_logp: np.ndarray,
    values: np.ndarray,
    packed_ids: list[np.ndarray],
    packed_offsets: list[np.ndarray],
    packed_meta: list[np.ndarray],
    mask_steps: list[np.ndarray],
    bootstrap_obs: np.ndarray,
    bootstrap_actor: np.ndarray,
    bootstrap_value: np.ndarray,
    initial_hidden_state: np.ndarray,
    final_hidden_state: np.ndarray,
    episode_seed: np.ndarray,
    policy_train_mask: np.ndarray,
    opponent_context_index: np.ndarray,
    teacher_family: np.ndarray | None,
    teacher_slot: np.ndarray | None,
    teacher_move_source: np.ndarray | None,
    teacher_attack_type: np.ndarray | None,
    teacher_action: np.ndarray | None,
    teacher_valid: np.ndarray | None,
    trajectory_retention_valid: np.ndarray | None,
    counters: dict[str, int],
    copy_counters: bool,
) -> RuntimeUnroll:
    bootstrap_arrays = normalize_collector_bootstrap_arrays(
        bootstrap_obs=bootstrap_obs,
        bootstrap_actor=bootstrap_actor,
        bootstrap_value=bootstrap_value,
    )
    counters["copied_bytes_estimate"] += estimate_collector_copied_bytes(
        obs,
        actions,
        rewards,
        terminated,
        truncated,
        to_play_seat,
        behavior_logp,
        values,
        episode_seed,
        policy_train_mask,
        opponent_context_index,
        teacher_family,
        teacher_slot,
        teacher_move_source,
        teacher_attack_type,
        teacher_action,
        teacher_valid,
        trajectory_retention_valid,
        bootstrap_arrays.obs,
        bootstrap_arrays.actor,
        bootstrap_arrays.value,
    )
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed),
        obs=obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        to_play_seat=to_play_seat,
        behavior_logp=behavior_logp,
        values=values,
        legal_actions=legal_actions_from_collector_steps(
            layout_name=layout_name,
            action_dim=action_dim,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
            mask_steps=mask_steps,
        ),
        bootstrap_obs=bootstrap_arrays.obs,
        bootstrap_actor=bootstrap_arrays.actor,
        bootstrap_value=bootstrap_arrays.value,
        initial_hidden_state=initial_hidden_state,
        final_hidden_state=np.asarray(final_hidden_state).copy(),
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
        counters=dict(counters) if copy_counters else counters,
    )
