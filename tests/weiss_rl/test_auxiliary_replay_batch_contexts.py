from __future__ import annotations

from typing import Any

import numpy as np
import torch
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_runner import AuxiliaryReplayBatchContext, run_auxiliary_replay_updates
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

from .auxiliary_replay_runner_test_support import (
    ContextModel,
    Learner,
    PlainModel,
    dataset_with_opponents,
)


def test_auxiliary_replay_runner_samples_context_batches_and_metrics() -> None:
    replay_dataset = dataset_with_opponents(["opponent_a", "opponent_b", "opponent_c", "opponent_d"])
    state = TrajectoryBcReplayState(
        dataset=replay_dataset,
        rng=np.random.default_rng(5),
        batch_episodes=2,
        aux_updates=2,
        every_updates=1,
        order=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    model = ContextModel({"opponent_a": 7, "opponent_b": 8, "opponent_c": 9, "opponent_d": 10})
    contexts: list[AuxiliaryReplayBatchContext] = []

    def update_batch(batch: dict[str, Any], context: AuxiliaryReplayBatchContext) -> dict[str, float]:
        contexts.append(context)
        assert context.opponent_context_indices is not None
        assert np.asarray(batch["initial_hidden_state"]).shape == (2, 3)
        assert np.asarray(batch["opponent_context_index"]).tolist() == [context.opponent_context_indices.tolist()]
        return {"loss": 0.25}

    result = run_auxiliary_replay_updates(
        sampler=state,
        learner=Learner(model=model),
        device=torch.device("cpu"),
        update_batch=update_batch,
        use_opponent_context=True,
    )
    latest: dict[str, float] = {}
    result.sampled_metrics.emit_common_metrics(latest, prefix="aux", include_context=True)

    assert [context.episode_indices for context in contexts] == [[0, 1], [2, 3]]
    assert [
        context.opponent_context_indices.tolist()
        for context in contexts
        if context.opponent_context_indices is not None
    ] == [
        [7, 8],
        [9, 10],
    ]
    assert model.initial_contexts == [[7, 8], [9, 10]]
    assert result.aux_metrics == {"loss": 0.25}
    assert latest["aux_aux_updates"] == 2.0
    assert latest["aux_batch_episodes"] == 4.0
    assert latest["aux_opponent_context_episodes"] == 4.0


def test_auxiliary_replay_runner_keeps_plain_replay_batches_context_free() -> None:
    replay_dataset = dataset_with_opponents(["opponent_a", "opponent_b"])
    state = TrajectoryBcReplayState(
        dataset=replay_dataset,
        rng=np.random.default_rng(5),
        batch_episodes=2,
        aux_updates=1,
        every_updates=1,
        order=np.asarray([0, 1], dtype=np.int64),
    )
    model = PlainModel()

    def batch_factory(
        dataset: ReplayTrajectoryDataset,
        *,
        episode_indices: list[int],
        initial_hidden_state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        batch = replay_trajectory_bc_batch(
            dataset,
            episode_indices=episode_indices,
            initial_hidden_state=initial_hidden_state,
        )
        assert "opponent_context_index" not in batch
        return batch

    def update_batch(batch: dict[str, Any], context: AuxiliaryReplayBatchContext) -> dict[str, float]:
        assert context.episode_indices == [0, 1]
        assert context.opponent_context_indices is None
        assert np.asarray(batch["initial_hidden_state"]).shape == (2, 2)
        return {"acc": 1.0}

    result = run_auxiliary_replay_updates(
        sampler=state,
        learner=Learner(model=model),
        device=torch.device("cpu"),
        update_batch=update_batch,
        batch_factory=batch_factory,
        use_opponent_context=False,
    )

    assert model.initial_batch_sizes == [2]
    assert result.aux_metrics == {"acc": 1.0}
