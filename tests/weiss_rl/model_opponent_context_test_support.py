from __future__ import annotations

from types import SimpleNamespace

import torch
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.model import build_policy_value_model


def opponent_context_config(
    *,
    context_scale: float = 0.0,
    trainable_context_scale: float = 0.0,
    trainable_recurrent_scale: float = 0.0,
    trainable_action_bias_scale: float = 0.0,
    trainable_candidate_residual_scale: float = 0.0,
    candidate_residual_mode: str = "additive",
    candidate_residual_action_ids: tuple[int, ...] = (),
) -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=16,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
        opponent_context_policy_ids=("B2 HeuristicPublic", "seed_c3aac2f9dc_policy_000004"),
        opponent_context_hidden_scale=context_scale,
        opponent_context_trainable_hidden_scale=trainable_context_scale,
        opponent_context_trainable_recurrent_scale=trainable_recurrent_scale,
        opponent_context_trainable_action_bias_scale=trainable_action_bias_scale,
        opponent_context_trainable_candidate_residual_scale=trainable_candidate_residual_scale,
        opponent_context_candidate_residual_mode=candidate_residual_mode,
        opponent_context_candidate_residual_action_ids=candidate_residual_action_ids,
        opponent_context_eval_policy_ids=("policy_000001",),
    )


def opponent_context_model(*, action_dim: int = 5, **config_kwargs):
    return build_policy_value_model(
        observation_dim=4,
        config=opponent_context_config(**config_kwargs),
        action_dim=action_dim,
    )


def candidate_residual_actions(*, meta: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ids=torch.tensor([104, 124, 104, 124], dtype=torch.long),
        offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        meta=meta
        if meta is not None
        else torch.tensor(
            [
                [2, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [2, 2, 0],
            ],
            dtype=torch.long,
        ),
    )
