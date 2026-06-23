from __future__ import annotations

from weiss_rl.replay.trajectory_bc import (
    merge_replay_trajectory_bc_datasets,
    replay_trajectory_bc_batch,
)

from .replay_trajectory_bc_dataset_test_support import synthetic_replay_trajectory_dataset


def test_merge_replay_trajectory_bc_datasets_offsets_preference_pair_ids() -> None:
    first = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
        ],
        train_mask=[[True, True]],
        label="first",
    )
    first.metadata["selected_bundles"] = [
        {"source_dataset_label": "first_preferred", "preference_pair_id": 0, "preference_role": 1},
        {"source_dataset_label": "first_rejected", "preference_pair_id": 0, "preference_role": 0},
    ]
    second = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
        ],
        train_mask=[[True, True]],
        label="second",
    )
    second.metadata["selected_bundles"] = [
        {"source_dataset_label": "second_preferred", "preference_pair_id": 0, "preference_role": 1},
        {"source_dataset_label": "second_rejected", "preference_pair_id": 0, "preference_role": 0},
    ]

    merged = merge_replay_trajectory_bc_datasets(
        [first, second],
        source_labels=("first", "second"),
        preserve_source_bundle_labels=True,
    )
    batch = replay_trajectory_bc_batch(merged, episode_indices=[0, 1, 2, 3])

    assert batch["preference_pair_id"].tolist() == [[0, 0, 1, 1]]
    assert [bundle["merge_source_preference_pair_id"] for bundle in merged.metadata["selected_bundles"]] == [
        0,
        0,
        0,
        0,
    ]
