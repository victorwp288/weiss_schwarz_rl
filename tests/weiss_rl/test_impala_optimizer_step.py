from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
from weiss_rl.learners.impala.updates.optimizer_step import run_impala_optimizer_step

from .impala_test_support import FakeGradScaler, ImpalaLearner, TinyPolicyValueModel


def test_run_impala_optimizer_step_reports_no_grad_without_stepping() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=torch.tensor(2.0),
        base_metrics={"loss": 2.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["loss"] == pytest.approx(2.0)
    assert metrics["optimizer_no_grad"] == pytest.approx(1.0)
    assert metrics["amp_grad_overflow"] == pytest.approx(0.0)
    assert metrics["loss_scale"] == pytest.approx(0.0)
    assert metrics["grad_norm"] == pytest.approx(0.0)


def test_run_impala_optimizer_step_preserves_standard_amp_backoff_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=True,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert np.isnan(metrics["grad_norm"])


def test_run_impala_optimizer_step_preserves_auxiliary_amp_update_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(8.0)
    assert np.isnan(metrics["grad_norm"])
