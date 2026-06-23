from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from weiss_rl.core.observation_layout import ObservationPlayerBlock, ObservationSlice, parse_observation_layout
from weiss_rl.model import (
    _block_segments,
    _flatten_indices,
    _TypedObservationEncoder,
    _TypedPlayerBlockEncoder,
    _TypedSegmentEncoder,
)
from weiss_rl.models.typed_encoder import (
    TypedObservationEncoder,
    TypedPlayerBlockEncoder,
    TypedSegmentEncoder,
    block_segments,
    build_observation_encoder,
    flatten_indices,
)


def _layout_spec() -> dict[str, object]:
    return {
        "obs_len": 8,
        "header_fields": [{"name": "turn", "index": 0}],
        "player_blocks": [
            {
                "name": "self",
                "base": 1,
                "len": 4,
                "slices": [
                    {"name": "stage", "start": 0, "len": 2},
                    {"name": "hand", "start": 2, "len": 2},
                ],
            }
        ],
        "tail_slices": [{"name": "global", "start": 5, "len": 3}],
    }


def test_typed_observation_encoder_preserves_output_shape_and_state_dict_names() -> None:
    layout = parse_observation_layout(_layout_spec())
    encoder = TypedObservationEncoder(
        layout=layout,
        feature_width=3,
        output_width=5,
        fusion_layers=1,
        layer_norm=False,
        dropout_p=0.0,
    )

    obs = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    assert encoder(obs).shape == (2, 5)
    assert "_header_encoder._projection.0.weight" in encoder.state_dict()
    assert "_player_encoders.0._slice_encoders.0._projection.0.weight" in encoder.state_dict()
    assert "_tail_encoder._projection.0.weight" in encoder.state_dict()


def test_typed_segment_encoder_rejects_empty_indices() -> None:
    with pytest.raises(ValueError, match="at least one observation index"):
        TypedSegmentEncoder(indices=(), output_width=4, layer_norm=False, dropout_p=0.0)


def test_typed_observation_encoder_requires_at_least_one_group() -> None:
    layout = parse_observation_layout({"obs_len": 3})

    with pytest.raises(ValueError, match="requires observation metadata"):
        TypedObservationEncoder(
            layout=layout,
            feature_width=3,
            output_width=5,
            fusion_layers=1,
            layer_norm=False,
            dropout_p=0.0,
        )


def test_typed_encoder_helper_aliases_are_preserved() -> None:
    block = ObservationPlayerBlock(name="empty", base=2, length=3, slices=())
    fallback = (ObservationSlice(name="empty_full", start=2, length=3),)
    slices = (ObservationSlice(name="a", start=1, length=2), ObservationSlice(name="b", start=4, length=1))

    assert _TypedObservationEncoder is TypedObservationEncoder
    assert _TypedPlayerBlockEncoder is TypedPlayerBlockEncoder
    assert _TypedSegmentEncoder is TypedSegmentEncoder
    assert _block_segments(block) == fallback
    assert block_segments(block) == fallback
    assert _flatten_indices(slices) == (1, 2, 4)
    assert flatten_indices(slices) == (1, 2, 4)


def test_build_observation_encoder_builds_mlp_and_typed_encoders() -> None:
    mlp_encoder = build_observation_encoder(
        observation_dim=4,
        config=SimpleNamespace(
            encoder_kind="mlp",
            encoder_mlp_width=6,
            encoder_mlp_layers=1,
            layer_norm=False,
            typed_feature_width=3,
        ),
        observation_spec=None,
        dropout_p=0.0,
        structured_encoder_kind="structured_v2",
    )
    assert mlp_encoder(torch.ones((2, 4))).shape == (2, 6)

    typed_encoder = build_observation_encoder(
        observation_dim=8,
        config=SimpleNamespace(
            encoder_kind="structured_v2",
            encoder_mlp_width=5,
            encoder_mlp_layers=1,
            layer_norm=False,
            typed_feature_width=3,
        ),
        observation_spec=_layout_spec(),
        dropout_p=0.0,
        structured_encoder_kind="structured_v2",
    )
    assert typed_encoder(torch.ones((2, 8))).shape == (2, 5)


def test_build_observation_encoder_preserves_error_messages() -> None:
    config = SimpleNamespace(
        encoder_kind="typed_v1",
        encoder_mlp_width=5,
        encoder_mlp_layers=1,
        layer_norm=False,
        typed_feature_width=3,
    )

    with pytest.raises(ValueError, match="typed_v1 encoder requires observation_spec"):
        build_observation_encoder(
            observation_dim=8,
            config=config,
            observation_spec=None,
            dropout_p=0.0,
            structured_encoder_kind="structured_v2",
        )

    with pytest.raises(ValueError, match="typed_v1 observation spec length mismatch"):
        build_observation_encoder(
            observation_dim=7,
            config=config,
            observation_spec=_layout_spec(),
            dropout_p=0.0,
            structured_encoder_kind="structured_v2",
        )

    unsupported_config = SimpleNamespace(
        encoder_kind="transformer",
        encoder_mlp_width=5,
        encoder_mlp_layers=1,
        layer_norm=False,
        typed_feature_width=3,
    )
    with pytest.raises(ValueError, match="Unsupported model.encoder_kind"):
        build_observation_encoder(
            observation_dim=8,
            config=unsupported_config,
            observation_spec=_layout_spec(),
            dropout_p=0.0,
            structured_encoder_kind="structured_v2",
        )
