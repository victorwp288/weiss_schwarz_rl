from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.losses.loss_objective_stage import compute_impala_objective_stage
from weiss_rl.learners.impala.losses.objective_loss import compute_impala_objective_losses

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel


def test_compute_impala_objective_stage_preserves_context_and_objective_contract() -> None:
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        trajectory_retention_coef=0.5,
        value_loss_coef=0.25,
        entropy_coef=0.1,
    )
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        obs=obs,
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        values=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        trajectory_retention_valid=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        factorized_result=SimpleNamespace(top_action_ids=torch.tensor([[0], [0]], dtype=torch.long)),
        context=context,
    )
    resolved_vtrace = SimpleNamespace(
        advantages=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        targets=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
    )
    policy_action_logp = torch.tensor([[-0.25], [-0.75]], dtype=torch.float32)
    retention_action_logp = torch.tensor([[-0.5], [-1.0]], dtype=torch.float32)
    entropy = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

    stage = compute_impala_objective_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        entropy=entropy,
        resolved_vtrace=resolved_vtrace,
        batch_value=lambda source, key: source.get(key),
    )
    direct = compute_impala_objective_losses(
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        actions=inputs.actions,
        advantages=resolved_vtrace.advantages,
        values=inputs.values,
        targets=resolved_vtrace.targets,
        entropy=entropy,
        loss_mask=inputs.loss_mask,
        value_loss_mask=context["value_train_mask"],
        value_loss_coef=float(learner.value_loss_coef),
        entropy_coef=float(learner.entropy_coef),
        trajectory_retention_valid=inputs.trajectory_retention_valid,
        trajectory_retention_coef=float(learner.trajectory_retention_coef),
        top_action_ids=inputs.factorized_result.top_action_ids,
    )

    torch.testing.assert_close(stage.losses.total_loss, direct.total_loss)
    torch.testing.assert_close(stage.losses.policy_loss, direct.policy_loss)
    torch.testing.assert_close(stage.losses.value_loss, direct.value_loss)
    torch.testing.assert_close(stage.losses.entropy_mean, direct.entropy_mean)
    torch.testing.assert_close(stage.losses.trajectory_retention_loss, direct.trajectory_retention_loss)
    assert stage.losses.trajectory_retention_metrics == pytest.approx(direct.trajectory_retention_metrics)
    assert context["value_train_mask"].requires_grad is False
    assert context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert context["trajectory_retention_loss"].requires_grad is False
