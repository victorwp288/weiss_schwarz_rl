from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.paired_swing_replay import PairedSwingReplayState

from .paired_swing_replay_test_support import build_training_config, dataset_with_action_pairs


def test_paired_swing_replay_state_reuses_grouped_sampler_and_counts_distinct_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_action_pairs(
            labels=["old", "fixed", "hardneg", "fixed", "old", "hardneg"],
            actions=[1, 1, 1, 1, 1, 1],
            teacher_actions=[1, 2, 2, 2, 1, 2],
        ),
    )

    state = PairedSwingReplayState.from_training_config(
        build_training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            focus_groups=(
                SimpleNamespace(name="fixed", source_labels=("fixed",), fraction=0.25),
                SimpleNamespace(name="hardneg", source_labels=("hardneg",), fraction=0.25),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    assert state.distinct_train_rows == 4
    indices = state.sampler.next_episode_indices()
    labels = [state.sampler.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 4
    assert sum(label == "fixed" for label in labels) == 1
    assert sum(label == "hardneg" for label in labels) == 1


def test_paired_swing_replay_rejects_dataset_without_distinct_action_pairs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_action_pairs(labels=["old", "old"], actions=[1, 1], teacher_actions=[1, 1]),
    )

    with pytest.raises(ValueError, match="no trainable rows where positive and negative actions differ"):
        PairedSwingReplayState.from_training_config(
            build_training_config(dataset_path=dataset_path),
            repo_root=tmp_path,
        )
