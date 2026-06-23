from __future__ import annotations

import numpy as np
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def synthetic_replay_trajectory_dataset(
    *,
    time_steps: int,
    episode_count: int,
    legal_rows: list[list[int]],
    train_mask: list[list[bool]],
    label: str,
    include_outcomes: list[str] | None = None,
) -> ReplayTrajectoryDataset:
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    legal_ids_parts: list[np.ndarray] = []
    legal_meta_parts: list[np.ndarray] = []
    offsets = [0]
    cursor = 0
    for row_ids in legal_rows:
        ids = np.asarray(row_ids, dtype=np.uint32)
        legal_ids_parts.append(ids)
        legal_meta_parts.append(np.stack([ids, ids + 10, ids + 20], axis=1).astype(np.uint16))
        cursor += int(ids.shape[0])
        offsets.append(cursor)
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=np.zeros((time_steps, episode_count), dtype=np.int8),
        to_play_seat=np.zeros((time_steps, episode_count), dtype=np.int8),
        actions=actions,
        legal_ids=np.concatenate(legal_ids_parts).astype(np.uint32),
        legal_offsets=np.asarray(offsets, dtype=np.uint32),
        legal_action_meta=np.concatenate(legal_meta_parts, axis=0).astype(np.uint16),
        teacher_family=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_slot=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_move_source=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_attack_type=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_action=np.full((time_steps, episode_count), -1, dtype=np.int32),
        teacher_valid=np.zeros((time_steps, episode_count), dtype=np.bool_),
        policy_train_mask=np.asarray(train_mask, dtype=np.bool_),
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": episode_count,
            "requested_bundle_count": episode_count,
            "include_outcomes": ["W"] if include_outcomes is None else include_outcomes,
            "pass_action_id": 0,
            "spec_hash256": "ab" * 32,
            "train_rows": int(np.count_nonzero(train_mask)),
            "selected_bundles": [{"source": label}],
        },
    )
