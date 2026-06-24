from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.losses.loss_objective_stage import resolve_impala_value_loss_mask
from weiss_rl.learners.impala.losses.loss_vtrace_stage import (
    attach_resolved_vtrace_context,
    resolve_impala_vtrace_clip_config,
)

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel


def test_resolve_impala_vtrace_clip_config_prefers_batch_overrides_then_learner_defaults() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=1.5, vtrace_c_bar=0.75)

    defaults = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={},
        batch_value=lambda source, key: source.get(key),
    )
    overrides = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={"vtrace_rho_bar": 2.25, "vtrace_c_bar": 0.5},
        batch_value=lambda source, key: source.get(key),
    )

    assert defaults.rho_bar == pytest.approx(1.5)
    assert defaults.c_bar == pytest.approx(0.75)
    assert overrides.rho_bar == pytest.approx(2.25)
    assert overrides.c_bar == pytest.approx(0.5)


def test_attach_resolved_vtrace_context_and_value_mask_keep_detached_loss_diagnostics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    values = torch.zeros((2, 1), dtype=torch.float32)
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    resolved_vtrace = SimpleNamespace(
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    context: dict[str, Any] = {}

    attach_resolved_vtrace_context(
        context=context,
        resolved_vtrace=resolved_vtrace,
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True),
    )
    value_mask = resolve_impala_value_loss_mask(
        learner=learner,
        batch=batch,
        expected_shape=torch.Size((2, 1)),
        like=values,
        batch_value=lambda source, key: source.get(key),
    )

    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False
    assert value_mask is not None
    assert value_mask.tolist() == [[0.0], [1.0]]
