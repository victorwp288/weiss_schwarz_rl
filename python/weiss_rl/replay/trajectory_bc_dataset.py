"""Replay trajectory BC dataset records, IO, and array operations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BC_DATASET_FORMAT = "weiss_rl_replay_trajectory_bc_v1"


@dataclass(frozen=True, slots=True)
class ReplayTrajectoryDataset:
    """Packed time-major replay supervision arrays plus JSON metadata."""

    obs: np.ndarray
    actor: np.ndarray
    to_play_seat: np.ndarray
    actions: np.ndarray
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray
    teacher_family: np.ndarray
    teacher_slot: np.ndarray
    teacher_move_source: np.ndarray
    teacher_attack_type: np.ndarray
    teacher_action: np.ndarray
    teacher_valid: np.ndarray
    policy_train_mask: np.ndarray
    reset_before_step: np.ndarray
    metadata: dict[str, Any]

    @property
    def time_steps(self) -> int:
        return int(self.obs.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.obs.shape[1])

    @property
    def row_count(self) -> int:
        return int(self.time_steps * self.episode_count)


def save_replay_trajectory_bc_dataset(path: Path, dataset: ReplayTrajectoryDataset) -> None:
    """Persist a replay trajectory BC dataset as a compressed npz artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=dataset.obs,
        actor=dataset.actor,
        to_play_seat=dataset.to_play_seat,
        actions=dataset.actions,
        legal_ids=dataset.legal_ids,
        legal_offsets=dataset.legal_offsets,
        legal_action_meta=dataset.legal_action_meta,
        teacher_family=dataset.teacher_family,
        teacher_slot=dataset.teacher_slot,
        teacher_move_source=dataset.teacher_move_source,
        teacher_attack_type=dataset.teacher_attack_type,
        teacher_action=dataset.teacher_action,
        teacher_valid=dataset.teacher_valid,
        policy_train_mask=dataset.policy_train_mask,
        reset_before_step=dataset.reset_before_step,
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True), dtype=np.str_),
    )


def load_replay_trajectory_bc_dataset(path: Path) -> ReplayTrajectoryDataset:
    """Load a compressed replay trajectory BC dataset."""

    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("format") != BC_DATASET_FORMAT:
            raise ValueError(f"Unsupported replay trajectory BC dataset format: {metadata.get('format')!r}")
        return ReplayTrajectoryDataset(
            obs=np.asarray(payload["obs"]),
            actor=np.asarray(payload["actor"]),
            to_play_seat=np.asarray(payload["to_play_seat"]),
            actions=np.asarray(payload["actions"]),
            legal_ids=np.asarray(payload["legal_ids"]),
            legal_offsets=np.asarray(payload["legal_offsets"]),
            legal_action_meta=np.asarray(payload["legal_action_meta"]),
            teacher_family=np.asarray(payload["teacher_family"]),
            teacher_slot=np.asarray(payload["teacher_slot"]),
            teacher_move_source=np.asarray(payload["teacher_move_source"]),
            teacher_attack_type=np.asarray(payload["teacher_attack_type"]),
            teacher_action=np.asarray(payload["teacher_action"]),
            teacher_valid=np.asarray(payload["teacher_valid"]),
            policy_train_mask=np.asarray(payload["policy_train_mask"]),
            reset_before_step=np.asarray(payload["reset_before_step"]),
            metadata=dict(metadata),
        )


def merge_replay_trajectory_bc_datasets(
    datasets: Sequence[ReplayTrajectoryDataset],
    *,
    source_labels: Sequence[str] | None = None,
    preserve_source_bundle_labels: bool = False,
    offset_preference_pair_ids: bool = True,
) -> ReplayTrajectoryDataset:
    """Concatenate trajectory BC datasets along the episode axis."""

    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("datasets must contain at least one dataset")
    labels = tuple(source_labels or ())
    if labels and len(labels) != len(dataset_list):
        raise ValueError("source_labels must match datasets length when provided")

    first = dataset_list[0]
    obs_dim = int(first.obs.shape[-1])
    pass_action_id = int(first.metadata.get("pass_action_id", -1))
    spec_hash = str(first.metadata.get("spec_hash256") or "")
    meta_width = int(first.legal_action_meta.shape[1]) if first.legal_action_meta.ndim == 2 else 3
    for index, dataset in enumerate(dataset_list):
        if int(dataset.obs.shape[-1]) != obs_dim:
            raise ValueError(f"dataset {index} obs_dim does not match first dataset")
        if int(dataset.metadata.get("pass_action_id", -1)) != pass_action_id:
            raise ValueError(f"dataset {index} pass_action_id does not match first dataset")
        if str(dataset.metadata.get("spec_hash256") or "") != spec_hash:
            raise ValueError(f"dataset {index} spec_hash256 does not match first dataset")
        if dataset.legal_action_meta.ndim != 2 or int(dataset.legal_action_meta.shape[1]) != meta_width:
            raise ValueError(f"dataset {index} legal_action_meta width does not match first dataset")

    max_time_steps = max(dataset.time_steps for dataset in dataset_list)
    episode_count = sum(dataset.episode_count for dataset in dataset_list)
    obs = np.zeros((max_time_steps, episode_count, obs_dim), dtype=np.float32)
    actor = np.zeros((max_time_steps, episode_count), dtype=np.int8)
    to_play_seat = np.zeros((max_time_steps, episode_count), dtype=np.int8)
    actions = np.full((max_time_steps, episode_count), pass_action_id, dtype=np.int64)
    teacher_family = np.full((max_time_steps, episode_count), -1, dtype=np.int32)
    teacher_slot = np.full((max_time_steps, episode_count), -1, dtype=np.int32)
    teacher_move_source = np.full((max_time_steps, episode_count), -1, dtype=np.int32)
    teacher_attack_type = np.full((max_time_steps, episode_count), -1, dtype=np.int32)
    teacher_action = np.full((max_time_steps, episode_count), -1, dtype=np.int32)
    teacher_valid = np.zeros((max_time_steps, episode_count), dtype=np.bool_)
    policy_train_mask = np.zeros((max_time_steps, episode_count), dtype=np.bool_)
    reset_before_step = np.zeros((max_time_steps, episode_count), dtype=np.bool_)

    padding_ids, padding_meta = _padding_row_from_dataset(first)
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    legal_offsets = [0]
    cursor = 0
    column_offsets = _column_offsets(dataset_list)
    for step_index in range(max_time_steps):
        for source_index, dataset in enumerate(dataset_list):
            column_offset = column_offsets[source_index]
            for episode_index in range(dataset.episode_count):
                target_column = column_offset + episode_index
                if step_index < dataset.time_steps:
                    obs[step_index, target_column] = dataset.obs[step_index, episode_index]
                    actor[step_index, target_column] = dataset.actor[step_index, episode_index]
                    to_play_seat[step_index, target_column] = dataset.to_play_seat[step_index, episode_index]
                    actions[step_index, target_column] = dataset.actions[step_index, episode_index]
                    teacher_family[step_index, target_column] = dataset.teacher_family[step_index, episode_index]
                    teacher_slot[step_index, target_column] = dataset.teacher_slot[step_index, episode_index]
                    teacher_move_source[step_index, target_column] = dataset.teacher_move_source[
                        step_index, episode_index
                    ]
                    teacher_attack_type[step_index, target_column] = dataset.teacher_attack_type[
                        step_index, episode_index
                    ]
                    teacher_action[step_index, target_column] = dataset.teacher_action[step_index, episode_index]
                    teacher_valid[step_index, target_column] = dataset.teacher_valid[step_index, episode_index]
                    policy_train_mask[step_index, target_column] = dataset.policy_train_mask[step_index, episode_index]
                    reset_before_step[step_index, target_column] = dataset.reset_before_step[step_index, episode_index]
                    row_ids, row_meta = _dataset_legal_row(dataset, step_index, episode_index)
                else:
                    row_ids, row_meta = padding_ids, padding_meta
                legal_ids_parts.append(row_ids)
                legal_meta_parts.append(row_meta)
                cursor += int(row_ids.shape[0])
                legal_offsets.append(cursor)

    legal_ids = np.concatenate(legal_ids_parts, axis=0).astype(np.uint32, copy=False)
    legal_action_meta = np.concatenate(legal_meta_parts, axis=0).astype(np.uint16, copy=False)
    metadata = {
        "format": BC_DATASET_FORMAT,
        "merged": True,
        "source_dataset_count": len(dataset_list),
        "source_datasets": _source_dataset_metadata(
            dataset_list,
            labels,
            preserve_source_bundle_labels=preserve_source_bundle_labels,
        ),
        "preserve_source_bundle_labels": bool(preserve_source_bundle_labels),
        "run_dir": str(first.metadata.get("run_dir", "")),
        "bundle_count": int(sum(int(dataset.metadata.get("bundle_count", 0)) for dataset in dataset_list)),
        "requested_bundle_count": int(
            sum(int(dataset.metadata.get("requested_bundle_count", 0)) for dataset in dataset_list)
        ),
        "include_outcomes": _merged_include_outcomes(dataset_list),
        "obs_dim": obs_dim,
        "time_steps": int(max_time_steps),
        "episode_count": int(episode_count),
        "row_count": int(max_time_steps * episode_count),
        "train_rows": int(np.count_nonzero(policy_train_mask)),
        "teacher_valid_rows": int(np.count_nonzero(teacher_valid)),
        "supported_target_rows": int(
            sum(int(dataset.metadata.get("supported_target_rows", 0)) for dataset in dataset_list)
        ),
        "unsupported_target_rows": int(
            sum(int(dataset.metadata.get("unsupported_target_rows", 0)) for dataset in dataset_list)
        ),
        "opponent_rows": int(sum(int(dataset.metadata.get("opponent_rows", 0)) for dataset in dataset_list)),
        "nonfocal_rows": int(sum(int(dataset.metadata.get("nonfocal_rows", 0)) for dataset in dataset_list)),
        "pass_action_id": int(pass_action_id),
        "spec_hash256": spec_hash,
        "selected_bundles": _selected_bundle_metadata(
            dataset_list,
            labels,
            preserve_source_bundle_labels=preserve_source_bundle_labels,
            offset_preference_pair_ids=offset_preference_pair_ids,
        ),
    }
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=np.asarray(legal_offsets, dtype=np.uint32),
        legal_action_meta=legal_action_meta,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_move_source=teacher_move_source,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        policy_train_mask=policy_train_mask,
        reset_before_step=reset_before_step,
        metadata=metadata,
    )


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


def _merged_include_outcomes(datasets: Sequence[ReplayTrajectoryDataset]) -> list[str]:
    values: set[str] = set()
    for dataset in datasets:
        raw = dataset.metadata.get("include_outcomes", [])
        if not raw:
            return []
        values.update(str(outcome) for outcome in raw)
    return sorted(values)


def _column_offsets(datasets: Sequence[ReplayTrajectoryDataset]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for dataset in datasets:
        offsets.append(cursor)
        cursor += dataset.episode_count
    return offsets


def _dataset_legal_row(
    dataset: ReplayTrajectoryDataset,
    step_index: int,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_index = int(step_index * dataset.episode_count + episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    return (
        np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32),
        np.asarray(dataset.legal_action_meta[start:stop], dtype=np.uint16),
    )


def _padding_row_from_dataset(dataset: ReplayTrajectoryDataset) -> tuple[np.ndarray, np.ndarray]:
    pass_action_id = int(dataset.metadata.get("pass_action_id", -1))
    if pass_action_id < 0:
        raise ValueError("dataset metadata is missing pass_action_id")
    for row_index in range(int(dataset.legal_offsets.shape[0]) - 1):
        start = int(dataset.legal_offsets[row_index])
        stop = int(dataset.legal_offsets[row_index + 1])
        legal_ids = np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32)
        pass_positions = np.flatnonzero(legal_ids.astype(np.int64, copy=False) == pass_action_id)
        if pass_positions.size:
            meta_row = np.asarray(dataset.legal_action_meta[start + int(pass_positions[0])], dtype=np.uint16)
            return (
                np.asarray([pass_action_id], dtype=np.uint32),
                meta_row.reshape(1, -1).astype(np.uint16, copy=False),
            )
    raise ValueError("could not derive pass padding metadata from dataset legal rows")


def _source_dataset_metadata(
    datasets: Sequence[ReplayTrajectoryDataset],
    labels: Sequence[str],
    *,
    preserve_source_bundle_labels: bool = False,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets):
        item = {
            "index": int(index),
            "label": labels[index] if labels else f"dataset_{index}",
            "bundle_count": int(dataset.metadata.get("bundle_count", 0)),
            "episode_count": int(dataset.episode_count),
            "time_steps": int(dataset.time_steps),
            "train_rows": int(dataset.metadata.get("train_rows", np.count_nonzero(dataset.policy_train_mask))),
            "path": str(dataset.metadata.get("dataset_path", "")),
        }
        if preserve_source_bundle_labels:
            item["preserve_source_bundle_labels"] = True
            nested_sources = dataset.metadata.get("source_datasets")
            if isinstance(nested_sources, list):
                item["nested_source_datasets"] = nested_sources
        payload.append(item)
    return payload


def _selected_bundle_metadata(
    datasets: Sequence[ReplayTrajectoryDataset],
    labels: Sequence[str],
    *,
    preserve_source_bundle_labels: bool = False,
    offset_preference_pair_ids: bool = True,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    preference_pair_offset = 0
    for index, dataset in enumerate(datasets):
        label = labels[index] if labels else f"dataset_{index}"
        source_pair_ids: list[int] = []
        for bundle in dataset.metadata.get("selected_bundles", []):
            if isinstance(bundle, Mapping):
                annotated = dict(bundle)
                raw_pair_id = annotated.get("preference_pair_id")
                if raw_pair_id is not None:
                    source_pair_id = int(raw_pair_id)
                    source_pair_ids.append(source_pair_id)
                    if offset_preference_pair_ids:
                        annotated["merge_source_preference_pair_id"] = source_pair_id
                        annotated["preference_pair_id"] = int(preference_pair_offset + source_pair_id)
                if preserve_source_bundle_labels and str(annotated.get("source_dataset_label") or ""):
                    annotated["merge_source_dataset_index"] = int(index)
                    annotated["merge_source_dataset_label"] = label
                else:
                    annotated.pop("merge_source_dataset_index", None)
                    annotated.pop("merge_source_dataset_label", None)
                    annotated["source_dataset_index"] = int(index)
                    annotated["source_dataset_label"] = label
                payload.append(annotated)
        if source_pair_ids:
            preference_pair_offset += max(source_pair_ids) + 1
    return payload


__all__ = [
    "BC_DATASET_FORMAT",
    "ReplayTrajectoryDataset",
    "load_replay_trajectory_bc_dataset",
    "merge_replay_trajectory_bc_datasets",
    "replay_trajectory_bc_batch",
    "save_replay_trajectory_bc_dataset",
    "subset_replay_trajectory_bc_dataset",
]
