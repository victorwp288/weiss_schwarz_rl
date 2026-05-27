from __future__ import annotations

import numpy as np

from weiss_rl.experiments.paired_swing_conflicts import (
    PairedSwingConflictConfig,
    build_paired_swing_conflict_report,
)
from weiss_rl.replay.trajectory_bc import (
    BC_DATASET_FORMAT,
    ReplayTrajectoryDataset,
    save_replay_trajectory_bc_dataset,
)


def test_paired_swing_conflict_report_detects_exact_reversed_same_history(tmp_path):
    dataset_a = _dataset_with_preference(
        positive_action=124,
        negative_action=104,
        source_label="fixed_preserve",
    )
    dataset_b = _dataset_with_preference(
        positive_action=104,
        negative_action=124,
        source_label="learned_repair",
    )
    path_a = tmp_path / "fixed.npz"
    path_b = tmp_path / "learned.npz"
    save_replay_trajectory_bc_dataset(path_a, dataset_a)
    save_replay_trajectory_bc_dataset(path_b, dataset_b)

    report = build_paired_swing_conflict_report(
        PairedSwingConflictConfig(dataset_paths=(path_a, path_b), max_examples=10)
    )

    assert report["preference_row_count"] == 2
    assert report["current_state_conflict_count"] == 1
    assert report["history_conflict_count"] == 1
    conflict = report["history_conflicts"][0]
    assert conflict["positive_actions"] == [104, 124]
    assert conflict["exact_reverse_pair_count"] == 1
    labels = {example["source_dataset_label"] for example in conflict["examples"]}
    assert labels == {"fixed_preserve", "learned_repair"}


def _dataset_with_preference(
    *, positive_action: int, negative_action: int, source_label: str
) -> ReplayTrajectoryDataset:
    obs = np.asarray([[[1.0, 0.0]], [[0.5, 0.5]]], dtype=np.float32)
    actor = np.asarray([[0], [0]], dtype=np.int64)
    to_play_seat = np.asarray([[0], [0]], dtype=np.int64)
    actions = np.asarray([[0], [positive_action]], dtype=np.int64)
    teacher_action = np.asarray([[-1], [negative_action]], dtype=np.int32)
    teacher_valid = np.asarray([[False], [True]], dtype=np.bool_)
    policy_train_mask = np.asarray([[False], [True]], dtype=np.bool_)
    legal_ids = np.asarray([0, 1, 104, 124], dtype=np.uint32)
    legal_offsets = np.asarray([0, 2, 4], dtype=np.uint32)
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=np.zeros((4, 4), dtype=np.uint16),
        teacher_family=np.full((2, 1), -1, dtype=np.int32),
        teacher_slot=np.full((2, 1), -1, dtype=np.int32),
        teacher_move_source=np.full((2, 1), -1, dtype=np.int32),
        teacher_attack_type=np.full((2, 1), -1, dtype=np.int32),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        policy_train_mask=policy_train_mask,
        reset_before_step=np.asarray([[True], [False]], dtype=np.bool_),
        metadata={
            "format": BC_DATASET_FORMAT,
            "train_rows": 1,
            "bundle_count": 1,
            "selected_bundles": [
                {
                    "source_dataset_label": source_label,
                    "source_pair_indices": [205],
                    "episode_seed": 14210367516666939508,
                }
            ],
        },
    )
