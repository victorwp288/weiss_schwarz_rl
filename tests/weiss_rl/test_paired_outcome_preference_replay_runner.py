from __future__ import annotations

import numpy as np
import torch
from weiss_rl.training.replay_data.paired_outcome_preference_replay import (
    PairedOutcomePreferenceReplayState,
    maybe_run_paired_outcome_preference_replay,
)

from .auxiliary_replay_runner_test_support import ContextModel, sampler


class PairedOutcomePreferenceLearner:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def paired_outcome_preference_update(self, batch: dict[str, object], **kwargs: object) -> dict[str, float]:
        self.calls.append({"batch": batch, "kwargs": dict(kwargs)})
        return {"preference_loss": 0.75}


def test_paired_outcome_preference_replay_preserves_groups_kwargs_context_and_metrics() -> None:
    replay_sampler = sampler(["opponent_b", "opponent_a"], batch_episodes=2, aux_updates=1, every_updates=1)
    state = PairedOutcomePreferenceReplayState(
        sampler=replay_sampler,
        beta=0.3,
        coef=0.7,
        aggregation="sum",
        group_balance=True,
        complete_pair_count=1,
    )
    learner = PairedOutcomePreferenceLearner(
        model=ContextModel({"opponent_a": 6, "opponent_b": 5}),
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_outcome_preference_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert len(learner.calls) == 1
    call = learner.calls[0]
    assert call["kwargs"] == {
        "beta": 0.3,
        "coef": 0.7,
        "aggregation": "sum",
        "group_balance": True,
    }
    assert np.asarray(call["batch"]["opponent_context_index"]).tolist() == [[5, 6]]
    assert np.asarray(call["batch"]["preference_group_id"]).tolist() == [[0, 1]]
    assert latest_metrics["paired_outcome_preference_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_outcome_preference_replay_opponent_context_episodes"] == 2.0
    assert latest_metrics["paired_outcome_preference_replay_complete_pair_count"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_beta"] == 0.3
    assert latest_metrics["paired_outcome_preference_replay_coef"] == 0.7
    assert latest_metrics["paired_outcome_preference_replay_aggregation_sum"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_group_balance"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_preference_loss"] == 0.75
