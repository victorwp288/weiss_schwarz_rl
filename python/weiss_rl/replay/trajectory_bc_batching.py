"""Episode-column slicing helpers for replay trajectory BC datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weiss_rl.replay.trajectory_bc_dataset_schema import ReplayTrajectoryDataset


def replay_trajectory_bc_batch(
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: Sequence[int],
    initial_hidden_state: np.ndarray | None = None,
    opponent_context_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Slice episode columns and rebuild packed legal offsets for a learner batch."""

    if not episode_indices:
        raise ValueError("episode_indices must contain at least one index")
    indices = np.asarray([int(index) for index in episode_indices], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= dataset.episode_count):
        raise ValueError("episode_indices contains an out-of-range episode index")
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    offsets = [0]
    total = 0
    original_batch = dataset.episode_count
    for step_index in range(dataset.time_steps):
        for episode_index in indices.tolist():
            row_index = int(step_index * original_batch + episode_index)
            start = int(dataset.legal_offsets[row_index])
            stop = int(dataset.legal_offsets[row_index + 1])
            row_ids = np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32)
            row_meta = np.asarray(dataset.legal_action_meta[start:stop], dtype=np.uint16)
            legal_ids_parts.append(row_ids)
            legal_meta_parts.append(row_meta)
            total += int(row_ids.shape[0])
            offsets.append(total)

    legal_ids = (
        np.concatenate(legal_ids_parts, axis=0).astype(np.uint32, copy=False)
        if legal_ids_parts
        else np.zeros((0,), dtype=np.uint32)
    )
    meta_width = int(dataset.legal_action_meta.shape[1]) if dataset.legal_action_meta.ndim == 2 else 3
    legal_action_meta = (
        np.concatenate(legal_meta_parts, axis=0).astype(np.uint16, copy=False)
        if legal_meta_parts
        else np.zeros((0, meta_width), dtype=np.uint16)
    )
    batch: dict[str, Any] = {
        "obs": np.asarray(dataset.obs[:, indices], dtype=np.float32),
        "actor": np.asarray(dataset.actor[:, indices], dtype=np.int64),
        "to_play_seat": np.asarray(dataset.to_play_seat[:, indices], dtype=np.int64),
        "actions": np.asarray(dataset.actions[:, indices], dtype=np.int64),
        "legal_ids": legal_ids,
        "legal_offsets": np.asarray(offsets, dtype=np.uint32),
        "legal_action_meta": legal_action_meta,
        "teacher_family": np.asarray(dataset.teacher_family[:, indices], dtype=np.int32),
        "teacher_slot": np.asarray(dataset.teacher_slot[:, indices], dtype=np.int32),
        "teacher_move_source": np.asarray(dataset.teacher_move_source[:, indices], dtype=np.int32),
        "teacher_attack_type": np.asarray(dataset.teacher_attack_type[:, indices], dtype=np.int32),
        "teacher_action": np.asarray(dataset.teacher_action[:, indices], dtype=np.int32),
        "teacher_valid": np.asarray(dataset.teacher_valid[:, indices], dtype=np.bool_),
        "policy_train_mask": np.asarray(dataset.policy_train_mask[:, indices], dtype=np.bool_),
        "reset_before_step": np.asarray(dataset.reset_before_step[:, indices], dtype=np.bool_),
    }
    source_label_ids = _source_label_ids_by_episode(dataset)
    if source_label_ids is not None:
        selected_label_ids = np.asarray(source_label_ids[indices], dtype=np.int64)
        batch["source_label_id"] = np.broadcast_to(
            selected_label_ids.reshape(1, -1),
            (int(dataset.time_steps), int(indices.shape[0])),
        ).copy()
    preference_pair_ids = _metadata_ints_by_episode(dataset, field_name="preference_pair_id")
    preference_roles = _metadata_ints_by_episode(dataset, field_name="preference_role")
    if preference_pair_ids is not None and preference_roles is not None:
        selected_pair_ids = np.asarray(preference_pair_ids[indices], dtype=np.int64)
        selected_roles = np.asarray(preference_roles[indices], dtype=np.int64)
        batch["preference_pair_id"] = np.broadcast_to(
            selected_pair_ids.reshape(1, -1),
            (int(dataset.time_steps), int(indices.shape[0])),
        ).copy()
        batch["preference_role"] = np.broadcast_to(
            selected_roles.reshape(1, -1),
            (int(dataset.time_steps), int(indices.shape[0])),
        ).copy()
    if initial_hidden_state is not None:
        batch["initial_hidden_state"] = np.asarray(initial_hidden_state)
    if opponent_context_indices is not None:
        context_indices = np.asarray(opponent_context_indices, dtype=np.int64).reshape(-1)
        if int(context_indices.shape[0]) != int(indices.shape[0]):
            raise ValueError(
                "opponent_context_indices must match selected episode count: "
                f"expected {int(indices.shape[0])}, got {int(context_indices.shape[0])}"
            )
        batch["opponent_context_index"] = np.broadcast_to(
            context_indices.reshape(1, -1),
            (int(dataset.time_steps), int(indices.shape[0])),
        ).copy()
    return batch


def subset_replay_trajectory_bc_dataset(
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: Sequence[int],
    selected_bundles: Sequence[Mapping[str, Any]] | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
) -> ReplayTrajectoryDataset:
    """Return an episode-column subset with rebuilt packed legal offsets."""

    batch = replay_trajectory_bc_batch(dataset, episode_indices=episode_indices)
    indices = [int(index) for index in episode_indices]
    raw_bundles = dataset.metadata.get("selected_bundles")
    if selected_bundles is None:
        if isinstance(raw_bundles, list):
            selected = [
                dict(raw_bundles[index]) if isinstance(raw_bundles[index], Mapping) else {} for index in indices
            ]
        else:
            selected = []
    else:
        selected = [dict(item) for item in selected_bundles]
    metadata = dict(dataset.metadata)
    metadata["bundle_count"] = len(indices)
    metadata["episode_count"] = len(indices)
    metadata["requested_bundle_count"] = len(indices)
    metadata["row_count"] = int(batch["obs"].shape[0] * batch["obs"].shape[1])
    metadata["time_steps"] = int(batch["obs"].shape[0])
    metadata["train_rows"] = int(np.count_nonzero(batch["policy_train_mask"]))
    metadata["teacher_valid_rows"] = int(np.count_nonzero(batch["teacher_valid"]))
    metadata["teacher_action_override_rows"] = int(np.count_nonzero(batch["policy_train_mask"]))
    metadata["selected_bundles"] = selected
    if metadata_updates:
        metadata.update(dict(metadata_updates))
    return ReplayTrajectoryDataset(
        obs=np.asarray(batch["obs"], dtype=np.float32),
        actor=np.asarray(batch["actor"]),
        to_play_seat=np.asarray(batch["to_play_seat"]),
        actions=np.asarray(batch["actions"]),
        legal_ids=np.asarray(batch["legal_ids"], dtype=np.uint32),
        legal_offsets=np.asarray(batch["legal_offsets"], dtype=np.uint32),
        legal_action_meta=np.asarray(batch["legal_action_meta"], dtype=np.uint16),
        teacher_family=np.asarray(batch["teacher_family"], dtype=np.int32),
        teacher_slot=np.asarray(batch["teacher_slot"], dtype=np.int32),
        teacher_move_source=np.asarray(batch["teacher_move_source"], dtype=np.int32),
        teacher_attack_type=np.asarray(batch["teacher_attack_type"], dtype=np.int32),
        teacher_action=np.asarray(batch["teacher_action"], dtype=np.int32),
        teacher_valid=np.asarray(batch["teacher_valid"], dtype=np.bool_),
        policy_train_mask=np.asarray(batch["policy_train_mask"], dtype=np.bool_),
        reset_before_step=np.asarray(batch["reset_before_step"], dtype=np.bool_),
        metadata=metadata,
    )


def _source_label_ids_by_episode(dataset: ReplayTrajectoryDataset) -> np.ndarray | None:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return None
    label_to_id: dict[str, int] = {}
    label_ids: list[int] = []
    for bundle in bundles:
        label = str(bundle.get("source_dataset_label") or "") if isinstance(bundle, Mapping) else ""
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        label_ids.append(label_to_id[label])
    return np.asarray(label_ids, dtype=np.int64)


def _metadata_ints_by_episode(dataset: ReplayTrajectoryDataset, *, field_name: str) -> np.ndarray | None:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return None
    values: list[int] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping) or field_name not in bundle:
            return None
        try:
            values.append(int(bundle[field_name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selected_bundles.{field_name} must be integer-like") from exc
    return np.asarray(values, dtype=np.int64)


__all__ = [
    "replay_trajectory_bc_batch",
    "subset_replay_trajectory_bc_dataset",
]
