"""Typed observation encoder modules used by policy/value models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from weiss_rl.core.observation_layout import (
    ObservationLayout,
    ObservationPlayerBlock,
    ObservationSlice,
    parse_observation_layout,
)
from weiss_rl.models.layers import build_mlp_stack


class TypedSegmentEncoder(nn.Module):
    def __init__(
        self,
        *,
        indices: Sequence[int],
        output_width: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if not indices:
            raise ValueError("Typed encoder segments must include at least one observation index")
        self.register_buffer("_indices", torch.as_tensor(tuple(int(index) for index in indices), dtype=torch.long))
        self._projection = build_mlp_stack(
            input_dim=len(indices),
            width=output_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self._projection(obs.index_select(1, cast(Tensor, self._indices)))


class TypedPlayerBlockEncoder(nn.Module):
    def __init__(
        self,
        *,
        block: ObservationPlayerBlock,
        feature_width: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        slice_encoders = [
            TypedSegmentEncoder(
                indices=current.indices,
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for current in block_segments(block)
        ]
        self._slice_encoders = nn.ModuleList(slice_encoders)
        self._fusion = build_mlp_stack(
            input_dim=len(slice_encoders) * feature_width,
            width=feature_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        encoded = [encoder(obs) for encoder in self._slice_encoders]
        return self._fusion(torch.cat(encoded, dim=1))


class TypedObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        layout: ObservationLayout,
        feature_width: int,
        output_width: int,
        fusion_layers: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if feature_width <= 0:
            raise ValueError(f"typed_feature_width must be >= 1, got {feature_width}")
        self._header_encoder = (
            TypedSegmentEncoder(
                indices=tuple(field.index for field in layout.header_fields),
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            if layout.header_fields
            else None
        )
        self._player_encoders = nn.ModuleList(
            TypedPlayerBlockEncoder(
                block=block,
                feature_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for block in layout.player_blocks
        )
        self._tail_encoder = (
            TypedSegmentEncoder(
                indices=flatten_indices(layout.tail_slices),
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            if layout.tail_slices
            else None
        )
        group_count = (
            len(self._player_encoders)
            + (0 if self._header_encoder is None else 1)
            + (0 if self._tail_encoder is None else 1)
        )
        if group_count == 0:
            raise ValueError(
                "typed_v1 encoder requires observation metadata with header_fields, player_blocks, or tail_slices"
            )
        self._fusion = build_mlp_stack(
            input_dim=group_count * feature_width,
            width=output_width,
            layers=fusion_layers,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        encoded_groups: list[Tensor] = []
        if self._header_encoder is not None:
            encoded_groups.append(self._header_encoder(obs))
        encoded_groups.extend(encoder(obs) for encoder in self._player_encoders)
        if self._tail_encoder is not None:
            encoded_groups.append(self._tail_encoder(obs))
        return self._fusion(torch.cat(encoded_groups, dim=1))


def block_segments(block: ObservationPlayerBlock) -> tuple[ObservationSlice, ...]:
    if block.slices:
        return block.slices
    return (ObservationSlice(name=f"{block.name}_full", start=block.base, length=block.length),)


def flatten_indices(slices: Sequence[ObservationSlice]) -> tuple[int, ...]:
    indices: list[int] = []
    for current in slices:
        indices.extend(current.indices)
    return tuple(indices)


def build_observation_encoder(
    *,
    observation_dim: int,
    config: Any,
    observation_spec: Mapping[str, Any] | None,
    dropout_p: float,
    structured_encoder_kind: str,
) -> nn.Module:
    encoder_kind = str(config.encoder_kind).strip().lower()
    if encoder_kind == "mlp":
        return build_mlp_stack(
            input_dim=observation_dim,
            width=config.encoder_mlp_width,
            layers=config.encoder_mlp_layers,
            layer_norm=config.layer_norm,
            dropout_p=dropout_p,
        )
    if encoder_kind not in {"typed_v1", structured_encoder_kind}:
        raise ValueError(f"Unsupported model.encoder_kind: {config.encoder_kind!r}")
    if observation_spec is None:
        raise ValueError(f"{encoder_kind} encoder requires observation_spec from the simulator spec bundle")
    layout = parse_observation_layout(observation_spec)
    if layout.obs_len != observation_dim:
        raise ValueError(
            f"{encoder_kind} observation spec length mismatch: expected {observation_dim}, observed {layout.obs_len}"
        )
    return TypedObservationEncoder(
        layout=layout,
        feature_width=config.typed_feature_width,
        output_width=config.encoder_mlp_width,
        fusion_layers=config.encoder_mlp_layers,
        layer_norm=config.layer_norm,
        dropout_p=dropout_p,
    )
