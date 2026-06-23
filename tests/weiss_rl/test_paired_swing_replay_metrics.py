from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset
from weiss_rl.training.replay_data.paired_swing_replay import PairedSwingReplayState, maybe_run_paired_swing_replay

from .paired_swing_replay_test_support import (
    ContextModel,
    ReplayLearner,
    build_training_config,
    dataset_with_action_pairs,
)


def test_maybe_run_paired_swing_replay_reports_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_action_pairs(labels=["fixed", "hardneg"], actions=[1, 1], teacher_actions=[2, 2]),
    )
    state = PairedSwingReplayState.from_training_config(
        build_training_config(dataset_path=dataset_path, batch_episodes=2),
        repo_root=tmp_path,
    )
    assert state is not None
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=ReplayLearner(),
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert latest_metrics["paired_swing_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_swing_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_dataset_distinct_train_rows"] == 2.0
    assert latest_metrics["paired_swing_replay_loss_scope_episode_mean"] == 0.0
    assert latest_metrics["paired_swing_replay_loss"] == pytest.approx(0.125)
    assert latest_metrics["paired_swing_replay_paired_swing_rows"] == 2.0


def test_maybe_run_paired_swing_replay_threads_opponent_context(tmp_path: Path) -> None:
    dataset_path = tmp_path / "paired_swing_context.npz"
    save_replay_trajectory_bc_dataset(
        dataset_path,
        dataset_with_action_pairs(
            labels=["fixed", "hardneg"],
            actions=[1, 1],
            teacher_actions=[2, 2],
            opponents=["B2 HeuristicPublic", "seed_c3aac2f9dc_policy_000005"],
        ),
    )
    state = PairedSwingReplayState.from_training_config(
        build_training_config(dataset_path=dataset_path, batch_episodes=2),
        repo_root=tmp_path,
    )
    assert state is not None
    model = ContextModel()
    learner = ReplayLearner(model=model)
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert learner.last_batch is not None
    context = np.asarray(learner.last_batch["opponent_context_index"], dtype=np.int64)
    source_label_id = np.asarray(learner.last_batch["source_label_id"], dtype=np.int64)
    assert context.shape == (1, 2)
    assert sorted(context.reshape(-1).tolist()) == [3, 7]
    assert source_label_id.shape == (1, 2)
    assert sorted(source_label_id.reshape(-1).tolist()) == [0, 1]
    assert sorted(model.last_initial_context.tolist()) == [3, 7]
    assert latest_metrics["paired_swing_replay_opponent_context_episodes"] == 2.0
