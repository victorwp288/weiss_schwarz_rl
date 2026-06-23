from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import weiss_rl.training.replay_data.trajectory_bc_replay as trajectory_bc_replay
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.trajectory_bc_replay import maybe_run_trajectory_bc_replay
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

from .trajectory_bc_replay_test_support import (
    ReplayLearner,
    dataset_with_source_labels,
    trajectory_bc_training_config,
)


def test_trajectory_bc_replay_metrics_accumulate_focus_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_source_labels(["old", "old", "repair_a", "repair_b", "old", "repair_a", "old", "repair_b"]),
    )
    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            aux_updates=2,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        repo_root=tmp_path,
    )
    assert state is not None

    monkeypatch.setattr(
        trajectory_bc_replay,
        "replay_trajectory_bc_batch",
        lambda dataset, *, episode_indices, initial_hidden_state: {"episode_indices": tuple(episode_indices)},
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_trajectory_bc_replay(
        state=state,
        learner=ReplayLearner(),
        training_config=trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=4,
            aux_updates=2,
            focus_source_labels=("repair_a", "repair_b"),
            focus_fraction=0.5,
        ),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["trajectory_bc_replay_aux_updates"] == 2.0
    assert latest_metrics["trajectory_bc_replay_batch_episodes"] == 8.0
    assert latest_metrics["trajectory_bc_replay_focus_batch_episodes"] == 4.0
    assert latest_metrics["trajectory_bc_replay_nonfocus_batch_episodes"] == 4.0


def test_trajectory_bc_replay_metrics_accumulate_named_focus_group_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "trajectory_bc.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_source_labels(["old", "old", "learned", "fixed", "old", "learned", "fixed", "old"]),
    )
    focus_groups = (
        SimpleNamespace(name="learned_repair", source_labels=("learned",), fraction=0.25),
        SimpleNamespace(name="fixed_repair", source_labels=("fixed",), fraction=0.25),
    )
    state = TrajectoryBcReplayState.from_training_config(
        trajectory_bc_training_config(
            dataset_path=dataset_path, batch_episodes=8, aux_updates=2, focus_groups=focus_groups
        ),
        repo_root=tmp_path,
    )
    assert state is not None

    monkeypatch.setattr(
        trajectory_bc_replay,
        "replay_trajectory_bc_batch",
        lambda dataset, *, episode_indices, initial_hidden_state: {"episode_indices": tuple(episode_indices)},
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_trajectory_bc_replay(
        state=state,
        learner=ReplayLearner(),
        training_config=trajectory_bc_training_config(
            dataset_path=dataset_path,
            batch_episodes=8,
            aux_updates=2,
            focus_groups=focus_groups,
        ),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["trajectory_bc_replay_focus_group_count"] == 2.0
    assert latest_metrics["trajectory_bc_replay_focus_batch_episodes"] == 8.0
    assert latest_metrics["trajectory_bc_replay_focus_group_learned_repair_batch_episodes"] == 4.0
    assert latest_metrics["trajectory_bc_replay_focus_group_fixed_repair_batch_episodes"] == 4.0
