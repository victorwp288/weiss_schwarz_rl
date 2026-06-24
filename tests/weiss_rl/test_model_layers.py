from __future__ import annotations

import pytest
import torch
from torch import nn
from weiss_rl.model import _build_mlp_stack
from weiss_rl.models.backbone.layers import build_mlp_stack


def test_build_mlp_stack_preserves_layer_order_and_output_shape() -> None:
    stack = build_mlp_stack(input_dim=3, width=5, layers=2, layer_norm=True, dropout_p=0.25)

    assert [type(layer) for layer in stack] == [
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Dropout,
    ]
    assert stack[0].in_features == 3
    assert stack[0].out_features == 5
    assert stack[4].in_features == 5
    assert stack[4].out_features == 5
    assert stack(torch.ones(2, 3)).shape == (2, 5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0, "width": 4, "layers": 1, "layer_norm": False, "dropout_p": 0.0}, "input_dim"),
        ({"input_dim": 1, "width": 0, "layers": 1, "layer_norm": False, "dropout_p": 0.0}, "width"),
        ({"input_dim": 1, "width": 4, "layers": 0, "layer_norm": False, "dropout_p": 0.0}, "layers"),
        ({"input_dim": 1, "width": 4, "layers": 1, "layer_norm": False, "dropout_p": 1.0}, "dropout_p"),
    ],
)
def test_build_mlp_stack_rejects_invalid_dimensions(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_mlp_stack(**kwargs)


def test_model_private_mlp_stack_wrapper_is_preserved() -> None:
    assert _build_mlp_stack is build_mlp_stack
