from __future__ import annotations

import time
from typing import Any, cast

import pytest
import torch
from torch import nn
from weiss_rl.learners.impala.updates.update_bookkeeping import (
    begin_impala_update_scope,
    finalize_impala_update_scope,
    set_impala_model_train_mode,
)
from weiss_rl.learners.impala.updates.update_loop import ScopedOptimizerUpdateSpec, run_scoped_impala_optimizer_update
from weiss_rl.learners.impala.updates.update_loss_stage import build_scoped_impala_loss

from .impala_test_support import ForwardProxyModel, ImpalaLearner, TinyPolicyValueModel, _simple_training_batch


def test_finalize_impala_update_scope_merges_and_clears_profile_timers() -> None:
    learner = ImpalaLearner(profile_timers=True)
    cast(Any, learner)._active_timing_metrics = {"timer_custom_ms": 3.5}

    metrics = finalize_impala_update_scope(
        learner=learner,
        metrics={"loss": 1.0},
        started_at=time.perf_counter(),
    )

    assert metrics["loss"] == pytest.approx(1.0)
    assert metrics["timer_custom_ms"] == pytest.approx(3.5)
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_auxiliary_update_scope_preserves_update_count_and_training_metrics_policy() -> None:
    learner = ImpalaLearner(profile_timers=True)
    learner.update_count = 7

    scope = begin_impala_update_scope(
        learner=learner,
        batch=_simple_training_batch(),
        count_learner_update=False,
        include_training_metrics=False,
        checkpoint_on_interval=False,
    )

    assert learner.update_count == 7
    assert learner.total_samples_processed == 2
    assert scope.metrics == {}
    assert cast(Any, learner)._active_timing_metrics == {}


def test_set_impala_model_train_mode_sets_compiled_model_too() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    model.eval()
    compiled_model.eval()
    learner = ImpalaLearner(model=model, compiled_model=compiled_model)

    set_impala_model_train_mode(learner)

    assert model.training is True
    assert compiled_model.training is True


def test_build_scoped_impala_loss_sets_train_mode_times_and_preserves_outputs() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    learner = ImpalaLearner(model=model, compiled_model=compiled_model, profile_timers=True)
    model.eval()
    compiled_model.eval()
    timings: list[tuple[str, float]] = []
    cast(Any, learner)._record_timing_ms = lambda name, duration: timings.append((name, duration))
    loss = model.policy.weight.sum()
    metrics = {"custom_loss": 1.0}
    context = {"custom_context": torch.tensor(1.0)}
    calls: list[str] = []

    stage = build_scoped_impala_loss(
        learner=learner,
        loss_timer_name="learner_custom_loss",
        build_loss=lambda: (
            calls.append("loss") or loss,
            metrics,
            context,
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert compiled_model.training is True
    assert stage.loss is loss
    assert stage.metrics is metrics
    assert stage.context is context
    assert [name for name, _duration in timings] == ["learner_custom_loss"]
    assert timings[0][1] >= 0.0


def test_run_scoped_impala_optimizer_update_preserves_auxiliary_scope_and_timing_contract() -> None:
    model = nn.Linear(1, 1, bias=False)
    learner = ImpalaLearner(model=model, profile_timers=True)
    learner.update_count = 4
    model.eval()
    calls: list[str] = []

    metrics = run_scoped_impala_optimizer_update(
        learner=learner,
        batch=_simple_training_batch(),
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message="missing model",
            loss_timer_name="learner_custom_loss",
        ),
        build_loss=lambda: (
            calls.append("loss") or model.weight.sum(),
            {"custom_loss": 1.0},
            {"context": torch.tensor(1.0)},
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert learner.update_count == 4
    assert learner.total_samples_processed == 2
    assert metrics["custom_loss"] == pytest.approx(1.0)
    assert "grad_norm" in metrics
    assert metrics["timer_learner_custom_loss_ms"] >= 0.0
    assert metrics["timer_learner_backward_ms"] >= 0.0
    assert metrics["timer_learner_optimizer_ms"] >= 0.0
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_run_scoped_impala_optimizer_update_rejects_missing_model_before_loss_build() -> None:
    learner = ImpalaLearner(model=None)
    calls: list[str] = []

    with pytest.raises(ValueError, match="custom missing model"):
        run_scoped_impala_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
            spec=ScopedOptimizerUpdateSpec(
                missing_model_message="custom missing model",
                loss_timer_name="learner_custom_loss",
            ),
            build_loss=lambda: (
                calls.append("loss") or torch.tensor(1.0),
                {},
                {},
            ),
        )

    assert calls == []
    assert learner.update_count == 0
    assert learner.total_samples_processed == 0
