from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.losses.action_reductions import resolve_impala_action_reductions
from weiss_rl.learners.impala.losses.loss_core import compute_impala_loss_core
from weiss_rl.learners.impala.losses.loss_inputs import prepare_impala_loss_inputs
from weiss_rl.learners.impala.losses.loss_pipeline import compute_impala_loss_and_metrics_with_context

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel, _simple_training_batch


def test_compute_impala_loss_core_finalizes_vtrace_objective_context_and_metrics() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.25,
        value_loss_coef=1.0,
        entropy_coef=0.0,
    )
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["value_train_mask"] = np.asarray([[False], [True]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    reductions = resolve_impala_action_reductions(
        factorized_result=inputs.factorized_result,
        logits=inputs.logits,
        packed_logits=inputs.packed_logits,
        legal_mask=inputs.legal_mask,
        packed_legal=inputs.packed_legal,
        actions=inputs.actions,
        entropy_scope=learner.entropy_scope,
        pass_action_id=learner.pass_action_id,
        action_catalog=getattr(learner.model, "action_catalog", None),
        record_timing_ms=learner._record_timing_ms,
    )
    inputs.context["action_logp"] = reductions.action_logp.detach()
    inputs.context["entropy"] = reductions.entropy.detach()

    result = compute_impala_loss_core(
        learner=learner,
        batch=batch,
        inputs=inputs,
        action_logp=reductions.action_logp,
        entropy=reductions.entropy,
        batch_value=lambda source, key: source.get(key),
    )

    assert result.context is inputs.context
    assert "targets" in result.context
    assert "advantages" in result.context
    assert "vtrace_rhos" in result.context
    assert "rewards" in result.context
    assert "trajectory_retention_loss" in result.context
    assert result.context["policy_train_mask"].tolist() == [[1.0], [0.0]]
    assert result.context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert result.metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["value_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert result.metrics["trajectory_retention_weighted_loss"] > 0.0
    assert result.metrics["loss"] == pytest.approx(float(result.total_loss.detach()))


def test_compute_impala_loss_pipeline_records_action_reductions_and_core_context() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()

    loss, metrics, context = compute_impala_loss_and_metrics_with_context(
        learner=learner,
        batch=batch,
        batch_value=lambda source, key: source.get(key),
    )

    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert context["action_logp"].shape == torch.Size((2, 1))
    assert context["entropy"].shape == torch.Size((2, 1))
    assert "targets" in context
    assert "advantages" in context
    assert "vtrace_rhos" in context
    assert "policy_train_mask" in context
    assert not context["action_logp"].requires_grad
    assert not context["entropy"].requires_grad
    assert metrics["policy_train_fraction"] == pytest.approx(1.0)
