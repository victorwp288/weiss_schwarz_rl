from __future__ import annotations

import pytest
import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.model import build_policy_value_model
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat


def _config(
    *,
    trainable_context_scale: float,
    trainable_recurrent_scale: float = 0.0,
    trainable_action_bias_scale: float = 0.0,
) -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=8,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
        opponent_context_policy_ids=("B2 HeuristicPublic",),
        opponent_context_trainable_hidden_scale=trainable_context_scale,
        opponent_context_trainable_recurrent_scale=trainable_recurrent_scale,
        opponent_context_trainable_action_bias_scale=trainable_action_bias_scale,
    )


def test_context_compat_loader_allows_missing_zero_init_adapter() -> None:
    old_model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_context_scale=0.0),
        action_dim=5,
    )
    new_model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_context_scale=0.5, trainable_recurrent_scale=1.0, trainable_action_bias_scale=1.0),
        action_dim=5,
    )

    result = load_model_state_dict_with_context_compat(
        new_model,
        old_model.state_dict(),
        context="unit test",
    )

    assert sorted(result.missing_keys) == [
        "opponent_context_action_bias_adapter",
        "opponent_context_hidden_adapter",
        "opponent_context_recurrent_adapter",
    ]
    assert torch.allclose(
        new_model.opponent_context_hidden_adapter, torch.zeros_like(new_model.opponent_context_hidden_adapter)
    )
    assert torch.allclose(
        new_model.opponent_context_action_bias_adapter,
        torch.zeros_like(new_model.opponent_context_action_bias_adapter),
    )
    assert torch.allclose(
        new_model.opponent_context_recurrent_adapter,
        torch.zeros_like(new_model.opponent_context_recurrent_adapter),
    )


def test_context_compat_loader_still_rejects_real_missing_keys() -> None:
    model = build_policy_value_model(
        observation_dim=4,
        config=_config(trainable_context_scale=0.5),
        action_dim=5,
    )
    state_dict = dict(model.state_dict())
    state_dict.pop("policy_head.weight")

    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_state_dict_with_context_compat(model, state_dict, context="unit test")
