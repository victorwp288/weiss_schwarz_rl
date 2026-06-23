from __future__ import annotations

from weiss_rl.replay.trajectory_bc import merge_replay_trajectory_bc_datasets

from .replay_trajectory_bc_dataset_test_support import synthetic_replay_trajectory_dataset


def test_merge_replay_trajectory_bc_datasets_can_preserve_nested_source_labels() -> None:
    repair_a = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="repair_a",
    )
    repair_b = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="repair_b",
    )
    premerged = merge_replay_trajectory_bc_datasets(
        [repair_a, repair_b],
        source_labels=["repair_a", "repair_b"],
    )
    loss_state = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 3]],
        train_mask=[[True]],
        label="b1_lossstate",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [premerged, loss_state],
        source_labels=["winnerrepair_mix", "b1_lossstate"],
        preserve_source_bundle_labels=True,
    )

    labels = [bundle["source_dataset_label"] for bundle in merged.metadata["selected_bundles"]]
    assert labels == ["repair_a", "repair_b", "b1_lossstate"]
    assert merged.metadata["selected_bundles"][0]["merge_source_dataset_label"] == "winnerrepair_mix"
    assert "nested_source_datasets" in merged.metadata["source_datasets"][0]


def test_merge_replay_trajectory_bc_datasets_flattens_stale_merge_labels_by_default() -> None:
    premerged = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 1]],
        train_mask=[[True]],
        label="old_mix",
    )
    premerged.metadata["selected_bundles"][0]["source_dataset_label"] = "old_source"
    premerged.metadata["selected_bundles"][0]["merge_source_dataset_label"] = "old_nested_mix"
    fresh = synthetic_replay_trajectory_dataset(
        time_steps=1,
        episode_count=1,
        legal_rows=[[0, 2]],
        train_mask=[[True]],
        label="fresh",
    )

    merged = merge_replay_trajectory_bc_datasets(
        [premerged, fresh],
        source_labels=["fixed_protect", "learned_repair"],
    )

    bundles = merged.metadata["selected_bundles"]
    assert bundles[0]["source_dataset_label"] == "fixed_protect"
    assert bundles[1]["source_dataset_label"] == "learned_repair"
    assert "merge_source_dataset_label" not in bundles[0]
