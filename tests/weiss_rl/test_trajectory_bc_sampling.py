from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.trajectory_bc_sampling import (
    TrajectoryBcReplayState,
    focus_group_counts,
    source_labels_by_episode,
)

from .trajectory_bc_replay_test_support import dataset_with_source_labels, trajectory_bc_training_config


def test_trajectory_bc_replay_stratifies_focus_source_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_source_labels(["old", "old", "repair_a", "repair_b", "old", "repair_a"]),
    )

    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 4
    assert sum(label in {"repair_a", "repair_b"} for label in labels) == 2
    assert state.last_focus_episode_count == 2
    assert state.last_nonfocus_episode_count == 2


def test_trajectory_bc_replay_rejects_missing_focus_source_label(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(dataset_path, dataset_with_source_labels(["old", "repair_a"]))

    with pytest.raises(ValueError, match="trajectory BC focus source labels not found"):
        TrajectoryBcReplayState.from_training_config(
            trajectory_bc_training_config(
                dataset_path=dataset_path,
                focus_source_labels=("missing_repair",),
                focus_fraction=0.5,
            ),
            repo_root=tmp_path,
        )


def test_trajectory_bc_replay_reserves_named_focus_group_fractions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_source_labels(
            [
                "old",
                "learned_a",
                "fixed_a",
                "old",
                "learned_b",
                "fixed_b",
                "old",
                "old",
            ]
        ),
    )

    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            focus_groups=(
                SimpleNamespace(name="learned", source_labels=("learned_a", "learned_b"), fraction=0.25),
                SimpleNamespace(name="fixed", source_labels=("fixed_a", "fixed_b"), fraction=0.25),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 8
    assert sum(label in {"learned_a", "learned_b"} for label in labels) == 2
    assert sum(label in {"fixed_a", "fixed_b"} for label in labels) == 2
    assert sum(label == "old" for label in labels) == 4
    assert state.last_focus_episode_count == 4
    assert state.last_nonfocus_episode_count == 4
    assert [group.last_episode_count for group in state.focus_groups] == [2, 2]


def test_trajectory_bc_replay_grouped_focus_fills_batch_when_no_nonfocus(tmp_path: Path) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_source_labels(["learned_a", "learned_b", "fixed_a", "fixed_b"]),
    )

    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            focus_groups=(
                SimpleNamespace(name="learned", source_labels=("learned_a", "learned_b"), fraction=0.16),
                SimpleNamespace(name="fixed", source_labels=("fixed_a", "fixed_b"), fraction=0.16),
            ),
        ),
        repo_root=tmp_path,
    )

    assert state is not None
    indices = state.next_episode_indices()

    labels = [state.dataset.metadata["selected_bundles"][index]["source_dataset_label"] for index in indices]
    assert len(indices) == 8
    assert sum(label in {"learned_a", "learned_b"} for label in labels) == 4
    assert sum(label in {"fixed_a", "fixed_b"} for label in labels) == 4
    assert state.last_focus_episode_count == 8
    assert state.last_nonfocus_episode_count == 0
    assert [group.last_episode_count for group in state.focus_groups] == [4, 4]


def test_trajectory_bc_replay_focus_group_counts_distribute_tied_remainders() -> None:
    assert focus_group_counts(
        batch_size=16,
        target_focus_count=11,
        fractions=(0.20, 0.15, 0.15, 0.15),
    ) == (3, 3, 3, 2)
    assert focus_group_counts(
        batch_size=16,
        target_focus_count=12,
        fractions=(0.16, 0.12, 0.14, 0.14, 0.14),
    ) == (3, 2, 3, 2, 2)


def test_trajectory_bc_sampling_source_labels_fallback_for_malformed_metadata() -> None:
    dataset = dataset_with_source_labels(["old", "repair_a", "repair_b"])
    dataset.metadata["selected_bundles"] = [{"source_dataset_label": "old"}]

    assert source_labels_by_episode(dataset) == ["", "", ""]
