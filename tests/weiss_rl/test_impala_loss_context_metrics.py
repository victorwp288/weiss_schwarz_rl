from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import weiss_rl.learners.impala.loss_context_stage as impala_loss_context_stage
from weiss_rl.learners.impala.loss_context_stage import finalize_impala_loss_context_stage
from weiss_rl.learners.impala.loss_finalization import finalize_impala_loss_context

from .impala_test_support import _FiniteRecorder


def test_finalize_impala_loss_context_records_losses_and_finite_checks() -> None:
    learner = _FiniteRecorder()
    context: dict[str, Any] = {}
    factorized_result = SimpleNamespace(family_log_probs=torch.log_softmax(torch.ones((1, 1, 3)), dim=-1))
    policy_loss = torch.tensor(0.5)
    value_loss = torch.tensor(1.0)
    entropy_mean = torch.tensor(0.25)
    total_loss = torch.tensor(1.375)
    policy_anchor_loss = torch.tensor(0.125)

    finalize_impala_loss_context(
        learner=learner,
        batch={"batch": True},
        context=context,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        total_loss=total_loss,
        policy_anchor_loss=policy_anchor_loss,
        factorized_result=factorized_result,
    )

    assert context["policy_loss"] is not policy_loss
    torch.testing.assert_close(context["policy_loss"], policy_loss)
    torch.testing.assert_close(context["value_loss"], value_loss)
    torch.testing.assert_close(context["entropy_mean"], entropy_mean)
    torch.testing.assert_close(context["policy_anchor_loss"], policy_anchor_loss)
    torch.testing.assert_close(context["total_loss"], total_loss)
    torch.testing.assert_close(context["factorized_family_log_probs"], factorized_result.family_log_probs)
    assert [name for name, _tensor in learner.calls] == [
        "policy_loss",
        "value_loss",
        "entropy_mean",
        "total_loss",
    ]


def test_finalize_impala_loss_context_stage_maps_objective_anchor_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"context_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    factorized_result = SimpleNamespace(family_log_probs=torch.zeros((1, 1, 3), dtype=torch.float32))
    inputs = SimpleNamespace(
        context=context,
        factorized_result=factorized_result,
    )
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_anchor_loss = torch.tensor(0.25, dtype=torch.float32)
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_loss=policy_anchor_loss)
    captured: dict[str, Any] = {}

    def fake_finalize_impala_loss_context(**kwargs: Any) -> None:
        captured.update(kwargs)
        kwargs["context"]["finalized"] = torch.tensor(1.0)

    monkeypatch.setattr(
        impala_loss_context_stage,
        "finalize_impala_loss_context",
        fake_finalize_impala_loss_context,
    )

    result_context = finalize_impala_loss_context_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
    )

    assert result_context is context
    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["context"] is context
    assert captured["policy_loss"] is policy_loss
    assert captured["value_loss"] is value_loss
    assert captured["entropy_mean"] is entropy_mean
    assert captured["total_loss"] is total_loss
    assert captured["policy_anchor_loss"] is policy_anchor_loss
    assert captured["factorized_result"] is factorized_result
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))
    torch.testing.assert_close(context["finalized"], torch.tensor(1.0))
