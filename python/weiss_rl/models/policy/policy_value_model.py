"""Dense policy/value model implementation.

This is the small fallback model: observation encoder, recurrent core, flat
policy head, and value head. The structured thesis model subclasses it and
replaces only the policy head.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.models.backbone.base import GLOBAL_ACTION_SPACE_SIZE, PolicyValueModelBaseMixin


class PolicyValueModel(PolicyValueModelBaseMixin, nn.Module):
    """Dense recurrent actor-critic for flat action-catalog scoring."""

    def __init__(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
        dropout_p: float | None = None,
        observation_spec: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be >= 1, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")

        self.observation_dim = observation_dim
        self.hidden_size = config.gru_hidden_size
        self.action_dim = action_dim
        self.recurrent_core = str(config.recurrent_core).strip().lower()
        self._configure_opponent_context(config=config, action_dim=action_dim)

        encoder_dropout = config.dropout.family_a if dropout_p is None else dropout_p
        self.encoder = self._build_observation_encoder(
            observation_dim=observation_dim,
            config=config,
            observation_spec=observation_spec,
            dropout_p=encoder_dropout,
        )
        self._install_recurrent_trunk_and_heads(config=config, action_dim=action_dim)

    def _install_recurrent_trunk_and_heads(self, *, config: ModelConfig, action_dim: int) -> None:
        self.gru = (
            nn.GRU(input_size=config.encoder_mlp_width, hidden_size=config.gru_hidden_size, batch_first=True)
            if self.recurrent_core == "gru"
            else None
        )
        self.feedforward_core = (
            None
            if self.recurrent_core == "gru"
            else nn.Sequential(nn.Linear(config.encoder_mlp_width, config.gru_hidden_size), nn.ReLU())
        )
        self.policy_head = nn.Linear(config.gru_hidden_size, action_dim)
        self.value_head = nn.Linear(config.gru_hidden_size, 1)


PolicyValueModel.__module__ = "weiss_rl.model"

__all__ = ["PolicyValueModel"]
