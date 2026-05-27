"""Reusable torch layer builders for policy/value models."""

from __future__ import annotations

import torch.nn as nn


def build_mlp_stack(
    *,
    input_dim: int,
    width: int,
    layers: int,
    layer_norm: bool,
    dropout_p: float,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"encoder input_dim must be >= 1, got {input_dim}")
    if width <= 0:
        raise ValueError(f"encoder width must be >= 1, got {width}")
    if layers <= 0:
        raise ValueError(f"encoder layers must be >= 1, got {layers}")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

    modules: list[nn.Module] = []
    in_features = input_dim
    for _ in range(layers):
        modules.append(nn.Linear(in_features, width))
        if layer_norm:
            modules.append(nn.LayerNorm(width))
        modules.append(nn.ReLU())
        if dropout_p > 0.0:
            modules.append(nn.Dropout(p=dropout_p))
        in_features = width
    return nn.Sequential(*modules)
