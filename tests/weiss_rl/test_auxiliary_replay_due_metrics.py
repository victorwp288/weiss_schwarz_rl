from __future__ import annotations

import numpy as np
import torch
from weiss_rl.training.auxiliary_replay_metrics import AuxiliaryReplayMetricAccumulator
from weiss_rl.training.auxiliary_replay_runner import (
    AuxiliaryReplayRunResult,
    auxiliary_replay_sampler_is_due,
    auxiliary_replay_update_is_due,
    emit_auxiliary_replay_aux_metrics,
    emit_auxiliary_replay_run_metrics,
    emit_auxiliary_replay_sampled_metrics,
)
from weiss_rl.training.replay_data.paired_auxiliary_replay import (
    emit_paired_auxiliary_replay_metrics,
    run_due_paired_auxiliary_replay,
)

from .auxiliary_replay_runner_test_support import Learner, PlainModel, sampler


def test_auxiliary_replay_due_preserves_positive_modulo_cadence() -> None:
    assert not auxiliary_replay_update_is_due(every_updates=2, update_count=0)
    assert not auxiliary_replay_update_is_due(every_updates=2, update_count=1)
    assert auxiliary_replay_update_is_due(every_updates=2, update_count=2)
    assert auxiliary_replay_update_is_due(every_updates=2, update_count=4)
    replay_sampler = sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=2)
    assert not auxiliary_replay_sampler_is_due(replay_sampler, update_count=1)
    assert auxiliary_replay_sampler_is_due(replay_sampler, update_count=2)


def test_auxiliary_replay_emission_helpers_preserve_common_aux_and_precedence() -> None:
    replay_sampler = sampler(["opponent_a", "opponent_b"], batch_episodes=2, aux_updates=1, every_updates=1)
    sampled_metrics = AuxiliaryReplayMetricAccumulator(replay_sampler)
    sampled_metrics.record_sampled_episodes([0, 1])
    sampled_metrics.record_context_indices(np.asarray([0, 5], dtype=np.int64))
    replay_result = AuxiliaryReplayRunResult(
        aux_metrics={"loss": 0.25, "coef": 0.9, "nan_loss": float("nan"), "label": "ignored"},
        sampled_metrics=sampled_metrics,
    )
    latest = {"example_replay_coef": 0.1}

    emit_auxiliary_replay_sampled_metrics(
        latest,
        prefix="example_replay",
        replay_result=replay_result,
        include_context=True,
    )
    assert latest["example_replay_aux_updates"] == 1.0
    assert latest["example_replay_batch_episodes"] == 2.0
    assert latest["example_replay_opponent_context_episodes"] == 1.0
    assert latest["example_replay_coef"] == 0.1

    latest["example_replay_coef"] = 0.7
    emit_auxiliary_replay_aux_metrics(latest, prefix="example_replay", replay_result=replay_result)
    assert latest["example_replay_loss"] == 0.25
    assert latest["example_replay_coef"] == 0.9
    assert "example_replay_nan_loss" not in latest
    assert "example_replay_label" not in latest

    combined_latest: dict[str, float] = {}
    emit_auxiliary_replay_run_metrics(
        combined_latest,
        prefix="combined_replay",
        replay_result=replay_result,
        include_context=True,
    )
    assert combined_latest["combined_replay_loss"] == 0.25
    assert combined_latest["combined_replay_opponent_context_episodes"] == 1.0


def test_run_due_paired_auxiliary_replay_skips_before_due_without_resolving_updater() -> None:
    from types import SimpleNamespace

    state = SimpleNamespace(sampler=sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=2))

    result = run_due_paired_auxiliary_replay(
        state=state,
        learner=Learner(model=PlainModel()),
        device=torch.device("cpu"),
        update_count=1,
        updater_method_name="missing_update",
        updater_error_message="missing updater",
        make_update_batch=lambda _updater: (_ for _ in ()).throw(AssertionError("should not build updater")),
    )

    assert result is None


def test_emit_paired_auxiliary_replay_metrics_preserves_static_then_aux_precedence() -> None:
    replay_sampler = sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=1)
    sampled_metrics = AuxiliaryReplayMetricAccumulator(replay_sampler)
    sampled_metrics.record_sampled_episodes([0])
    replay_result = AuxiliaryReplayRunResult(
        aux_metrics={"loss": 0.25, "coef": 0.9},
        sampled_metrics=sampled_metrics,
    )
    latest: dict[str, float] = {}

    emit_paired_auxiliary_replay_metrics(
        latest,
        prefix="paired",
        replay_result=replay_result,
        static_metrics={"paired_coef": 0.1, "paired_static": 2.0},
        include_context=False,
    )

    assert latest["paired_aux_updates"] == 1.0
    assert latest["paired_batch_episodes"] == 1.0
    assert latest["paired_static"] == 2.0
    assert latest["paired_coef"] == 0.9
    assert latest["paired_loss"] == 0.25
