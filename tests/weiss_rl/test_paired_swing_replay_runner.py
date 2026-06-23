from __future__ import annotations

import numpy as np
import torch
from weiss_rl.training.replay_data.paired_swing_replay import PairedSwingReplayState, maybe_run_paired_swing_replay

from .auxiliary_replay_runner_test_support import ContextModel, sampler


class PairedSwingLearner:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def paired_swing_update(self, batch: dict[str, object], **kwargs: object) -> dict[str, float]:
        self.calls.append({"batch": batch, "kwargs": dict(kwargs)})
        return {"paired_swing_loss": 0.5}


def test_paired_swing_replay_preserves_updater_kwargs_context_batch_and_metrics() -> None:
    replay_sampler = sampler(["opponent_a", "opponent_b"], batch_episodes=2, aux_updates=1, every_updates=2)
    state = PairedSwingReplayState(
        sampler=replay_sampler,
        margin=0.4,
        coef=0.2,
        positive_action_source="teacher_action",
        negative_action_source="actions",
        distinct_train_rows=2,
        loss_scope="label_mean",
        compare_to="top_other",
    )
    learner = PairedSwingLearner(
        model=ContextModel({"opponent_a": 3, "opponent_b": 4}),
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )
    assert learner.calls == []
    assert latest_metrics == {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=2,
        latest_metrics=latest_metrics,
    )

    assert len(learner.calls) == 1
    call = learner.calls[0]
    assert call["kwargs"] == {
        "margin": 0.4,
        "coef": 0.2,
        "positive_action_source": "teacher_action",
        "negative_action_source": "actions",
        "loss_scope": "label_mean",
        "compare_to": "top_other",
    }
    assert np.asarray(call["batch"]["opponent_context_index"]).tolist() == [[3, 4]]
    assert latest_metrics["paired_swing_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_swing_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_opponent_context_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_dataset_distinct_train_rows"] == 2.0
    assert latest_metrics["paired_swing_replay_margin"] == 0.4
    assert latest_metrics["paired_swing_replay_coef"] == 0.2
    assert latest_metrics["paired_swing_replay_loss_scope_label_mean"] == 1.0
    assert latest_metrics["paired_swing_replay_compare_to_top_other"] == 1.0
    assert latest_metrics["paired_swing_replay_positive_source_teacher"] == 1.0
    assert latest_metrics["paired_swing_replay_paired_swing_loss"] == 0.5
