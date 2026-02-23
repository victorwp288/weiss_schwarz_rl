"""Model configuration and interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    gru_hidden_size: int = 256
    encoder_mlp_width: int = 256
    encoder_mlp_layers: int = 2
    layer_norm: bool = True


class PolicyValueModel:
    """Placeholder interface for the recurrent actor-critic model."""

    def forward(self, obs, hidden_state):
        raise NotImplementedError("Implement model forward pass")
