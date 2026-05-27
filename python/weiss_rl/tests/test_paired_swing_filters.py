from __future__ import annotations

from pathlib import Path

import numpy as np

from weiss_rl.experiments.paired_swing_filters import (
    PairedSwingEpisodeFilterConfig,
    filter_paired_swing_dataset,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, save_replay_trajectory_bc_dataset


def test_filter_paired_swing_dataset_keeps_source_pair_and_rebuilds_offsets(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "pair205.npz"
    save_replay_trajectory_bc_dataset(source_path, _dataset())

    filtered, summary = filter_paired_swing_dataset(
        PairedSwingEpisodeFilterConfig(
            dataset_path=source_path,
            output_dataset_path=output_path,
            source_pair_indices=(205,),
            positive_action_source="actions",
            negative_action_source="teacher_action",
        )
    )

    assert output_path.is_file()
    assert filtered.episode_count == 2
    assert filtered.metadata["train_rows"] == 2
    assert filtered.metadata["selected_bundles"][0]["source_pair_index"] == 205
    assert filtered.metadata["selected_bundles"][1]["source_pair_indices"] == [205]
    assert filtered.legal_offsets.tolist() == [0, 2, 4, 6, 8]
    assert filtered.legal_ids.tolist() == [1, 2, 3, 4, 1, 2, 3, 4]
    assert filtered.actions.tolist() == [[1, 3], [1, 3]]
    assert filtered.teacher_action.tolist() == [[2, 4], [-1, -1]]
    assert summary["kept_episode_indices"] == [0, 2]
    assert summary["distinct_train_rows"] == 2


def _dataset() -> ReplayTrajectoryDataset:
    return ReplayTrajectoryDataset(
        obs=np.zeros((2, 3, 4), dtype=np.float32),
        actor=np.zeros((2, 3), dtype=np.int8),
        to_play_seat=np.zeros((2, 3), dtype=np.int8),
        actions=np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64),
        legal_ids=np.asarray([1, 2, 2, 3, 3, 4, 1, 2, 2, 3, 3, 4], dtype=np.uint32),
        legal_offsets=np.asarray([0, 2, 4, 6, 8, 10, 12], dtype=np.uint32),
        legal_action_meta=np.zeros((12, 4), dtype=np.uint16),
        teacher_family=np.full((2, 3), -1, dtype=np.int32),
        teacher_slot=np.full((2, 3), -1, dtype=np.int32),
        teacher_move_source=np.full((2, 3), -1, dtype=np.int32),
        teacher_attack_type=np.full((2, 3), -1, dtype=np.int32),
        teacher_action=np.asarray([[2, 2, 4], [-1, -1, -1]], dtype=np.int32),
        teacher_valid=np.asarray([[True, True, True], [False, False, False]], dtype=np.bool_),
        policy_train_mask=np.asarray([[True, True, True], [False, False, False]], dtype=np.bool_),
        reset_before_step=np.zeros((2, 3), dtype=np.bool_),
        metadata={
            "format": "weiss_rl_replay_trajectory_bc_v1",
            "bundle_count": 3,
            "episode_count": 3,
            "time_steps": 2,
            "row_count": 6,
            "train_rows": 3,
            "selected_bundles": [
                {
                    "source_dataset_label": "b2",
                    "source_pair_index": 205,
                    "source_opponent_policy_id": "B2 HeuristicPublic",
                },
                {
                    "source_dataset_label": "p5",
                    "source_pair_index": 68,
                    "source_opponent_policy_id": "seed_policy_000005",
                },
                {
                    "source_dataset_label": "best",
                    "source_pair_indices": [205],
                    "source_opponent_policy_id": "seed_best",
                },
            ],
        },
    )
