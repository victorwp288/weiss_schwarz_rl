from __future__ import annotations

import pytest
import torch
from torch import nn

from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.impala.optimizer_step import optimizer_has_gradients
from weiss_rl.learners.impala.optimizer_support import ImpalaOptimizerSupportMixin
from weiss_rl.learners.impala.update_loop import _optimizer_has_gradients
from weiss_rl.model import build_policy_value_model


def test_impala_learner_uses_canonical_optimizer_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaOptimizerSupportMixin)


def test_impala_update_loop_reexports_optimizer_gradient_check_for_compatibility() -> None:
    assert _optimizer_has_gradients is optimizer_has_gradients


def test_optimizer_uses_adapter_lr_multiplier_for_trainable_opponent_context() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        action_dim=5,
        config=ModelConfig(
            gru_hidden_size=8,
            encoder_mlp_width=8,
            encoder_mlp_layers=1,
            layer_norm=False,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            opponent_context_trainable_hidden_scale=1.0,
            opponent_context_trainable_recurrent_scale=1.0,
            opponent_context_trainable_action_bias_scale=1.0,
            opponent_context_adapter_lr_multiplier=100.0,
        ),
    )
    learner = ImpalaLearner(model=model, learning_rate=0.001)

    optimizer = learner._optimizer_for_step()

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[1]["params"] == [
        model.opponent_context_hidden_adapter,
        model.opponent_context_recurrent_adapter,
        model.opponent_context_action_bias_adapter,
    ]


def test_optimizer_can_train_only_opponent_context_adapters() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        action_dim=5,
        config=ModelConfig(
            gru_hidden_size=8,
            encoder_mlp_width=8,
            encoder_mlp_layers=1,
            layer_norm=False,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            opponent_context_trainable_action_bias_scale=1.0,
            opponent_context_adapter_lr_multiplier=100.0,
            opponent_context_adapter_train_only=True,
        ),
    )
    learner = ImpalaLearner(model=model, learning_rate=0.001)

    optimizer = learner._optimizer_for_step()

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["params"] == [model.opponent_context_action_bias_adapter]
    assert model.opponent_context_action_bias_adapter.requires_grad is True
    assert model.policy_head.weight.requires_grad is False
    assert model.value_head.weight.requires_grad is False


def test_optimizer_can_train_only_candidate_residual_adapter_parameters() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        action_dim=5,
        config=ModelConfig(
            gru_hidden_size=8,
            encoder_mlp_width=8,
            encoder_mlp_layers=1,
            layer_norm=False,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            opponent_context_adapter_lr_multiplier=10.0,
            opponent_context_adapter_train_only=True,
        ),
    )
    model.opponent_context_candidate_residual_context = nn.Parameter(torch.zeros((2, 4)))
    model.opponent_context_candidate_residual_state = nn.Linear(4, 4, bias=False)
    model.opponent_context_candidate_residual_meta = nn.Linear(3, 4, bias=False)
    model.opponent_context_candidate_residual_out = nn.Linear(4, 1, bias=False)
    learner = ImpalaLearner(model=model, learning_rate=0.001)

    optimizer = learner._optimizer_for_step()

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert set(optimizer.param_groups[0]["params"]) == {
        model.opponent_context_candidate_residual_context,
        model.opponent_context_candidate_residual_state.weight,
        model.opponent_context_candidate_residual_meta.weight,
        model.opponent_context_candidate_residual_out.weight,
    }
    assert model.policy_head.weight.requires_grad is False


def test_optimizer_has_gradients_detects_no_grad_adapter_only_step() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        action_dim=5,
        config=ModelConfig(
            gru_hidden_size=8,
            encoder_mlp_width=8,
            encoder_mlp_layers=1,
            layer_norm=False,
            dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
            opponent_context_policy_ids=("B2 HeuristicPublic",),
            opponent_context_trainable_hidden_scale=1.0,
            opponent_context_adapter_train_only=True,
        ),
    )
    learner = ImpalaLearner(model=model, learning_rate=0.001)
    optimizer = learner._optimizer_for_step()
    optimizer.zero_grad(set_to_none=True)

    torch.ones((), requires_grad=True).backward()

    assert _optimizer_has_gradients(optimizer) is False
