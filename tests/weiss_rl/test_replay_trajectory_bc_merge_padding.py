from __future__ import annotations

from weiss_rl.replay.trajectory_bc import (
    merge_replay_trajectory_bc_datasets,
    replay_trajectory_bc_batch,
)

from .replay_trajectory_bc_dataset_test_support import synthetic_replay_trajectory_dataset


def test_merge_replay_trajectory_bc_datasets_pads_shorter_sources() -> None:
    first = synthetic_replay_trajectory_dataset(
        time_steps=2,
        episode_count=1,
        legal_rows=[
            [0, 1],
            [0, 2],
        ],
        train_mask=[[True], [False]],
        label="b4_win",
    )
    second = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 3],
            [0, 4],
        ],
        train_mask=[[True, True]],
        label="b2_win",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [first, second],
        source_labels=["b4_win", "b2_win"],
    )

    assert merged.obs.shape == (2, 3, 4)
    assert merged.metadata["source_dataset_count"] == 2
    assert merged.metadata["bundle_count"] == 3
    assert merged.metadata["train_rows"] == 3
    assert merged.metadata["source_datasets"][1]["label"] == "b2_win"
    assert merged.policy_train_mask.tolist() == [[True, True, True], [False, False, False]]
    assert merged.actions[:, 1:].tolist() == [[1, 1], [0, 0]]
    assert merged.legal_offsets.shape == (7,)
    padded_row_1_start = int(merged.legal_offsets[4])
    padded_row_1_stop = int(merged.legal_offsets[5])
    assert merged.legal_ids[padded_row_1_start:padded_row_1_stop].tolist() == [0]
    padded_row_2_start = int(merged.legal_offsets[5])
    padded_row_2_stop = int(merged.legal_offsets[6])
    assert merged.legal_ids[padded_row_2_start:padded_row_2_stop].tolist() == [0]

    batch = replay_trajectory_bc_batch(merged, episode_indices=[0, 2])
    assert batch["obs"].shape == (2, 2, 4)
    assert batch["legal_offsets"].tolist() == [0, 2, 4, 6, 7]


def test_merge_replay_trajectory_bc_datasets_preserves_all_outcome_sentinel() -> None:
    all_outcomes = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="all_outcomes",
        include_outcomes=[],
    )
    wins_only = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="wins_only",
        include_outcomes=["W"],
    )

    merged = merge_replay_trajectory_bc_datasets([all_outcomes, wins_only])

    assert merged.metadata["include_outcomes"] == []
