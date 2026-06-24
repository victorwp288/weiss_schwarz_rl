"""Storage contract for replay trajectory behavior-cloning datasets."""

from __future__ import annotations

import json
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


__all__ = [
    "BC_DATASET_FORMAT",
    "ReplayTrajectoryDataset",
    "load_replay_trajectory_bc_dataset",
    "save_replay_trajectory_bc_dataset",
]
