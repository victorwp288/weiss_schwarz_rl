from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.vtrace import VTraceTargets

from .impala_test_support import (
    ImpalaLearner,
    TinyPolicyValueModel,
)


def test_impala_learner_trains_value_on_non_policy_rows_by_default() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_train_fraction"] == pytest.approx(1.0)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert context["value_train_mask"].tolist() == [[1.0], [1.0]]


def test_impala_learner_accepts_explicit_value_train_mask() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "value_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["value_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_loss"] == pytest.approx(0.0)
    assert context["value_train_mask"].tolist() == [[1.0], [0.0]]
