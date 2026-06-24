"""Step-row encoding and recurrent collation for replay BC datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from weiss_rl.config import StackConfig
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.sampling.model_action_surface import (
    ModelActionSurfaceSettings,
    action_catalog_action_surface_batch_and_ids,
)
from weiss_rl.replay.bundles import ReplayStep
from weiss_rl.replay.rerun_validation import load_observation_layout
from weiss_rl.runtime.components.actions.legal_meta import legal_action_meta_from_ids
from weiss_rl.runtime.components.teacher_labels import teacher_labels_from_actions


@dataclass(slots=True)
class TrajectoryBcStepRow:
    obs: np.ndarray
    actor: int
    action: int
    legal_ids: np.ndarray
    legal_action_meta: np.ndarray
    teacher_family: int
    teacher_slot: int
    teacher_move_source: int
    teacher_attack_type: int
    teacher_action: int
    teacher_valid: bool
    policy_train: bool
    teacher_action_overridden: bool
    decision_kind: int
    supported_target: bool


def build_step_row(
    *,
    batch: DecisionBoundaryBatch,
    expected_step: ReplayStep,
    focal_seat: int,
    legal_ids: np.ndarray,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
    teacher_action_override: int | None,
    override_mode: bool,
) -> TrajectoryBcStepRow:
    actor = int(batch.actor[0])
    action = int(expected_step.action)
    teacher_action = action if teacher_action_override is None else int(teacher_action_override)
    legal_ids_array = np.asarray(legal_ids, dtype=np.uint32)
    legal_meta = _single_row_legal_meta(batch, expected_count=int(legal_ids_array.shape[0]))
    supported_target = bool(np.any(legal_ids_array.astype(np.int64, copy=False) == teacher_action))
    policy_train = bool(
        actor == int(focal_seat)
        and supported_target
        and (not bool(override_mode) or teacher_action_override is not None)
    )
    labels = teacher_labels_from_actions(
        row_indices=np.asarray([0], dtype=np.int64),
        chosen_actions=np.asarray([teacher_action], dtype=np.int64),
        num_rows=1,
        guidance_active=policy_train,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
    teacher_family, teacher_slot, teacher_move_source, teacher_attack_type, label_action, teacher_valid = labels
    return TrajectoryBcStepRow(
        obs=np.asarray(batch.obs[0], dtype=np.float32),
        actor=actor,
        action=action,
        legal_ids=legal_ids_array,
        legal_action_meta=legal_meta,
        teacher_family=int(teacher_family[0]),
        teacher_slot=int(teacher_slot[0]),
        teacher_move_source=int(teacher_move_source[0]),
        teacher_attack_type=int(teacher_attack_type[0]),
        teacher_action=int(label_action[0]),
        teacher_valid=bool(teacher_valid[0]),
        policy_train=policy_train,
        teacher_action_overridden=teacher_action_override is not None,
        decision_kind=int(np.asarray(batch.decision_kind, dtype=np.int32)[0]),
        supported_target=supported_target,
    )


def collate_episode_rows(
    episodes: Sequence[Sequence[TrajectoryBcStepRow]],
    *,
    pass_action_id: int,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> dict[str, np.ndarray]:
    time_steps = max(len(episode) for episode in episodes)
    episode_count = len(episodes)
    obs_dim = int(episodes[0][0].obs.shape[0])
    obs = np.zeros((time_steps, episode_count, obs_dim), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int8)
    actions = np.zeros((time_steps, episode_count), dtype=np.int64)
    teacher_family = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_slot = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_move_source = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_attack_type = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_action = np.full((time_steps, episode_count), -1, dtype=np.int32)
    teacher_valid = np.zeros((time_steps, episode_count), dtype=np.bool_)
    policy_train_mask = np.zeros((time_steps, episode_count), dtype=np.bool_)
    reset_before_step = np.zeros((time_steps, episode_count), dtype=np.bool_)

    padding_ids, padding_meta = _padding_legal_row(
        pass_action_id=pass_action_id,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
    )
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    legal_offsets = [0]
    cursor = 0
    for step_index in range(time_steps):
        for episode_index, episode in enumerate(episodes):
            if step_index < len(episode):
                row = episode[step_index]
                obs[step_index, episode_index] = row.obs
                actor[step_index, episode_index] = np.int8(row.actor)
                actions[step_index, episode_index] = int(row.action)
                teacher_family[step_index, episode_index] = row.teacher_family
                teacher_slot[step_index, episode_index] = row.teacher_slot
                teacher_move_source[step_index, episode_index] = row.teacher_move_source
                teacher_attack_type[step_index, episode_index] = row.teacher_attack_type
                teacher_action[step_index, episode_index] = row.teacher_action
                teacher_valid[step_index, episode_index] = row.teacher_valid
                policy_train_mask[step_index, episode_index] = row.policy_train
                row_ids = row.legal_ids
                row_meta = row.legal_action_meta
            else:
                row_ids = padding_ids
                row_meta = padding_meta
            legal_ids_parts.append(row_ids)
            legal_meta_parts.append(row_meta)
            cursor += int(row_ids.shape[0])
            legal_offsets.append(cursor)

    legal_ids = np.concatenate(legal_ids_parts, axis=0).astype(np.uint32, copy=False)
    legal_action_meta = np.concatenate(legal_meta_parts, axis=0).astype(np.uint16, copy=False)
    return {
        "obs": obs,
        "actor": actor,
        "to_play_seat": actor.astype(np.int8, copy=True),
        "actions": actions,
        "legal_ids": legal_ids,
        "legal_offsets": np.asarray(legal_offsets, dtype=np.uint32),
        "legal_action_meta": legal_action_meta,
        "teacher_family": teacher_family,
        "teacher_slot": teacher_slot,
        "teacher_move_source": teacher_move_source,
        "teacher_attack_type": teacher_attack_type,
        "teacher_action": teacher_action,
        "teacher_valid": teacher_valid,
        "policy_train_mask": policy_train_mask,
        "reset_before_step": reset_before_step,
    }


def batch_with_legal_meta(
    batch: DecisionBoundaryBatch,
    *,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> DecisionBoundaryBatch:
    if batch.ids_offsets is None:
        return batch
    legal_ids, _legal_offsets = batch.ids_offsets
    if batch.legal_action_meta is not None:
        return batch
    meta = legal_action_meta_from_ids(
        np.asarray(legal_ids, dtype=np.uint32),
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=3,
    )
    return replace(batch, legal_action_meta=meta)


def filter_training_action_surface(
    *,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    stack: StackConfig,
    action_catalog: ActionCatalog,
    run_spec_bundle: Mapping[str, Any],
    pass_action_id: int,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    settings = ModelActionSurfaceSettings.from_training_config(
        stack.config.training,
        pass_action_id=pass_action_id,
    )
    if not settings.has_guards:
        return batch, legal_ids
    return action_catalog_action_surface_batch_and_ids(
        action_catalog=action_catalog,
        observation_layout=load_observation_layout(run_spec_bundle),
        batch=batch,
        legal_ids=legal_ids,
        settings=settings,
    )


def _single_row_legal_meta(batch: DecisionBoundaryBatch, *, expected_count: int) -> np.ndarray:
    if batch.legal_action_meta is None:
        raise RuntimeError("Replay trajectory BC extraction requires legal action metadata")
    meta = np.asarray(batch.legal_action_meta, dtype=np.uint16)
    if meta.ndim != 2:
        raise RuntimeError("legal_action_meta must be a matrix")
    if meta.shape[0] != int(expected_count):
        raise RuntimeError(
            f"legal_action_meta row count must match filtered legal ids: expected {expected_count}, got {meta.shape[0]}"
        )
    return meta


def _padding_legal_row(
    *,
    pass_action_id: int,
    action_catalog: ActionCatalog,
    family_index: dict[str, int],
    attack_type_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([int(pass_action_id)], dtype=np.uint32)
    meta = legal_action_meta_from_ids(
        ids,
        action_catalog=action_catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=3,
    )
    if meta is None:
        raise RuntimeError("Could not build padding legal-action metadata")
    return ids, meta


__all__ = [
    "TrajectoryBcStepRow",
    "batch_with_legal_meta",
    "build_step_row",
    "collate_episode_rows",
    "filter_training_action_surface",
]
