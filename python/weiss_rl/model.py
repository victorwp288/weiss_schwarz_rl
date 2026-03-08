"""Model configuration and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    gru_hidden_size: int = 256
    encoder_mlp_width: int = 256
    encoder_mlp_layers: int = 2
    layer_norm: bool = True


class PolicyValueModel:
    """Base interface for the recurrent actor-critic model."""

    def forward(self, obs: Any, hidden_state: Any) -> tuple[Any, Any]:
        raise RuntimeError("PolicyValueModel.forward() must be overridden by a concrete model.")
