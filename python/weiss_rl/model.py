"""Torch recurrent actor-critic model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.observation_layout import ObservationLayout, ObservationPlayerBlock, ObservationSlice, parse_observation_layout

GLOBAL_ACTION_SPACE_SIZE = 527
SEAT_COUNT = 2


def _build_mlp_stack(
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


class _TypedSegmentEncoder(nn.Module):
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
        self._projection = _build_mlp_stack(
            input_dim=len(indices),
            width=output_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self._projection(obs.index_select(1, self._indices))


class _TypedPlayerBlockEncoder(nn.Module):
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
            _TypedSegmentEncoder(
                indices=current.indices,
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for current in _block_segments(block)
        ]
        self._slice_encoders = nn.ModuleList(slice_encoders)
        self._fusion = _build_mlp_stack(
            input_dim=len(slice_encoders) * feature_width,
            width=feature_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        encoded = [encoder(obs) for encoder in self._slice_encoders]
        return self._fusion(torch.cat(encoded, dim=1))


class _TypedObservationEncoder(nn.Module):
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
            _TypedSegmentEncoder(
                indices=tuple(field.index for field in layout.header_fields),
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            if layout.header_fields
            else None
        )
        self._player_encoders = nn.ModuleList(
            _TypedPlayerBlockEncoder(
                block=block,
                feature_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for block in layout.player_blocks
        )
        self._tail_encoder = (
            _TypedSegmentEncoder(
                indices=_flatten_indices(layout.tail_slices),
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
        self._fusion = _build_mlp_stack(
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


def _block_segments(block: ObservationPlayerBlock) -> tuple[ObservationSlice, ...]:
    if block.slices:
        return block.slices
    return (ObservationSlice(name=f"{block.name}_full", start=block.base, length=block.length),)


def _flatten_indices(slices: Sequence[ObservationSlice]) -> tuple[int, ...]:
    indices: list[int] = []
    for current in slices:
        indices.extend(current.indices)
    return tuple(indices)


class PolicyValueModel(nn.Module):
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

        encoder_dropout = config.dropout.family_a if dropout_p is None else dropout_p
        self.encoder = self._build_observation_encoder(
            observation_dim=observation_dim,
            config=config,
            observation_spec=observation_spec,
            dropout_p=encoder_dropout,
        )
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

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return torch.zeros(batch_size, self.hidden_size, device=hidden_device, dtype=hidden_dtype)

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return torch.zeros(batch_size, SEAT_COUNT, self.hidden_size, device=hidden_device, dtype=hidden_dtype)

    def encode(self, obs: Tensor) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        return self.encoder(obs_batch)

    def recurrent_step(self, encoded_obs: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        hidden_batch = self._prepare_hidden_state(hidden_state, batch_size=batch_size, like=encoded_obs)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            recurrent_output, next_hidden = self.gru(encoded_obs.unsqueeze(1), hidden_batch.unsqueeze(0))
            return recurrent_output[:, 0, :], next_hidden[0]
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), hidden_batch

    def recurrent_step_seat_aware(
        self,
        encoded_obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        seat_hidden_batch = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded_obs,
        )
        acting_seat_batch = self._prepare_acting_seat(acting_seat, batch_size=batch_size, device=encoded_obs.device)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            acting_hidden_batch = self._select_acting_hidden(seat_hidden_batch, acting_seat_batch)
            recurrent_output, next_acting_hidden = self.gru(encoded_obs.unsqueeze(1), acting_hidden_batch.unsqueeze(0))
            next_seat_hidden = self._write_acting_hidden(seat_hidden_batch, acting_seat_batch, next_acting_hidden[0])
            return recurrent_output[:, 0, :], next_seat_hidden
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), seat_hidden_batch

    def recurrent_step_seat_aware_inplace(
        self,
        encoded_obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        seat_hidden_batch = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded_obs,
        )
        acting_seat_batch = self._prepare_acting_seat(acting_seat, batch_size=batch_size, device=encoded_obs.device)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            acting_hidden_batch = self._select_acting_hidden(seat_hidden_batch, acting_seat_batch)
            recurrent_output, next_acting_hidden = self.gru(encoded_obs.unsqueeze(1), acting_hidden_batch.unsqueeze(0))
            next_hidden = next_acting_hidden[0]
            if next_hidden.dtype != seat_hidden_batch.dtype:
                next_hidden = next_hidden.to(dtype=seat_hidden_batch.dtype)
            batch_index = torch.arange(seat_hidden_batch.shape[0], device=seat_hidden_batch.device)
            seat_hidden_batch[batch_index, acting_seat_batch] = next_hidden
            return recurrent_output[:, 0, :], seat_hidden_batch
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), seat_hidden_batch

    def forward(self, obs: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden


    def _build_observation_encoder(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        observation_spec: Mapping[str, Any] | None,
        dropout_p: float,
    ) -> nn.Module:
        encoder_kind = str(config.encoder_kind).strip().lower()
        if encoder_kind == "mlp":
            return _build_mlp_stack(
                input_dim=observation_dim,
                width=config.encoder_mlp_width,
                layers=config.encoder_mlp_layers,
                layer_norm=config.layer_norm,
                dropout_p=dropout_p,
            )
        if encoder_kind != "typed_v1":
            raise ValueError(f"Unsupported model.encoder_kind: {config.encoder_kind!r}")
        if observation_spec is None:
            raise ValueError("typed_v1 encoder requires observation_spec from the simulator spec bundle")
        layout = parse_observation_layout(observation_spec)
        if layout.obs_len != observation_dim:
            raise ValueError(
                "typed_v1 observation spec length mismatch: "
                f"expected {observation_dim}, observed {layout.obs_len}"
            )
        return _TypedObservationEncoder(
            layout=layout,
            feature_width=config.typed_feature_width,
            output_width=config.encoder_mlp_width,
            fusion_layers=config.encoder_mlp_layers,
            layer_norm=config.layer_norm,
            dropout_p=dropout_p,
        )

    def _require_observation_batch(self, obs: Tensor) -> Tensor:
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2D (batch, observation), got shape {tuple(obs.shape)}")
        if obs.shape[1] != self.observation_dim:
            raise ValueError(f"obs feature dimension mismatch: expected {self.observation_dim}, got {obs.shape[1]}")
        return obs.to(dtype=self.policy_head.weight.dtype)

    def _hidden_tensor_device_dtype(
        self,
        *,
        batch_size: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        hidden_device: torch.device = self.policy_head.weight.device if device is None else device
        hidden_dtype: torch.dtype = self.policy_head.weight.dtype if dtype is None else dtype
        return hidden_device, hidden_dtype

    def _prepare_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        if hidden_state is None:
            return self.initial_hidden(batch_size, device=like.device, dtype=like.dtype)
        if hidden_state.ndim != 2:
            raise ValueError(f"hidden_state must be 2D (batch, hidden_size), got shape {tuple(hidden_state.shape)}")
        if hidden_state.shape[0] != batch_size:
            raise ValueError(f"hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
        if hidden_state.shape[1] != self.hidden_size:
            raise ValueError(f"hidden_state feature mismatch: expected {self.hidden_size}, got {hidden_state.shape[1]}")
        return hidden_state.to(device=like.device, dtype=like.dtype)

    def _prepare_seat_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        if hidden_state is None:
            return self.initial_seat_hidden(batch_size, device=like.device, dtype=like.dtype)
        if hidden_state.ndim != 3:
            raise ValueError(
                f"seat_hidden_state must be 3D (batch, seat, hidden_size), got shape {tuple(hidden_state.shape)}"
            )
        if hidden_state.shape[0] != batch_size:
            raise ValueError(f"seat_hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
        if hidden_state.shape[1] != SEAT_COUNT:
            raise ValueError(f"seat_hidden_state seat mismatch: expected {SEAT_COUNT}, got {hidden_state.shape[1]}")
        if hidden_state.shape[2] != self.hidden_size:
            raise ValueError(
                f"seat_hidden_state feature mismatch: expected {self.hidden_size}, got {hidden_state.shape[2]}"
            )
        return hidden_state.to(device=like.device, dtype=like.dtype)

    def _prepare_acting_seat(self, acting_seat: int | Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        if isinstance(acting_seat, int):
            seat_batch = torch.full((batch_size,), acting_seat, device=device, dtype=torch.long)
        else:
            if acting_seat.is_floating_point() or acting_seat.is_complex():
                raise ValueError("acting_seat must contain integer seat ids")
            if acting_seat.ndim == 0:
                seat_batch = acting_seat.to(device=device, dtype=torch.long).expand(batch_size)
            elif acting_seat.ndim == 1:
                if acting_seat.shape[0] != batch_size:
                    raise ValueError(f"acting_seat batch mismatch: expected {batch_size}, got {acting_seat.shape[0]}")
                seat_batch = acting_seat.to(device=device, dtype=torch.long)
            else:
                raise ValueError(f"acting_seat must be scalar or 1D [batch], got shape {tuple(acting_seat.shape)}")
        if not torch.all((seat_batch == 0) | (seat_batch == 1)):
            raise ValueError("acting_seat values must be 0 or 1")
        return seat_batch

    def _select_acting_hidden(self, seat_hidden_state: Tensor, acting_seat: Tensor) -> Tensor:
        acting_index = acting_seat.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return torch.gather(seat_hidden_state, dim=1, index=acting_index).squeeze(1)

    def _write_acting_hidden(
        self,
        seat_hidden_state: Tensor,
        acting_seat: Tensor,
        next_acting_hidden: Tensor,
    ) -> Tensor:
        next_seat_hidden = seat_hidden_state.clone()
        if next_acting_hidden.dtype != next_seat_hidden.dtype:
            next_acting_hidden = next_acting_hidden.to(dtype=next_seat_hidden.dtype)
        batch_index = torch.arange(seat_hidden_state.shape[0], device=seat_hidden_state.device)
        next_seat_hidden[batch_index, acting_seat] = next_acting_hidden
        return next_seat_hidden


__all__ = ["GLOBAL_ACTION_SPACE_SIZE", "SEAT_COUNT", "ModelConfig", "PolicyValueModel"]
