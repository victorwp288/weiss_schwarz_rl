from __future__ import annotations

from weiss_rl.replay.trajectory_bc import replay_trajectory_bc_batch

from .replay_trajectory_bc_dataset_test_support import synthetic_replay_trajectory_dataset


def test_replay_trajectory_bc_batch_broadcasts_preference_metadata() -> None:
    dataset = synthetic_replay_trajectory_dataset(
        time_steps=2,
        episode_count=2,
        legal_rows=[
            [0, 1],
            [0, 1],
            [0],
            [0],
        ],
        train_mask=[[True, True], [False, False]],
        label="preference",
    )
    dataset.metadata["selected_bundles"] = [
        {"source_dataset_label": "preferred", "preference_pair_id": 42, "preference_role": 1},
        {"source_dataset_label": "rejected", "preference_pair_id": 42, "preference_role": 0},
    ]

    batch = replay_trajectory_bc_batch(dataset, episode_indices=[1, 0])

    assert batch["preference_pair_id"].tolist() == [[42, 42], [42, 42]]
    assert batch["preference_role"].tolist() == [[0, 1], [0, 1]]
    assert batch["source_label_id"].tolist() == [[1, 0], [1, 0]]
