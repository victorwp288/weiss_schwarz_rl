"""Torch recurrent actor-critic model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig

GLOBAL_ACTION_SPACE_SIZE = 527
SEAT_COUNT = 2


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
        dropout_p: float | None = None,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be >= 1, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")

        self.observation_dim = observation_dim
        self.hidden_size = config.gru_hidden_size
        self.action_dim = action_dim

        encoder_dropout = config.dropout.family_a if dropout_p is None else dropout_p
        self.encoder = self._build_encoder(
            input_dim=observation_dim,
            width=config.encoder_mlp_width,
            layers=config.encoder_mlp_layers,
            layer_norm=config.layer_norm,
            dropout_p=encoder_dropout,
        )
        self.gru = nn.GRU(input_size=config.encoder_mlp_width, hidden_size=config.gru_hidden_size, batch_first=True)
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
        recurrent_output, next_hidden = self.gru(encoded_obs.unsqueeze(1), hidden_batch.unsqueeze(0))
        return recurrent_output[:, 0, :], next_hidden[0]

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
        acting_hidden_batch = self._select_acting_hidden(seat_hidden_batch, acting_seat_batch)
        recurrent_output, next_acting_hidden = self.gru(encoded_obs.unsqueeze(1), acting_hidden_batch.unsqueeze(0))
        next_seat_hidden = self._write_acting_hidden(seat_hidden_batch, acting_seat_batch, next_acting_hidden[0])
        return recurrent_output[:, 0, :], next_seat_hidden

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

    @staticmethod
    def _build_encoder(
        *,
        input_dim: int,
        width: int,
        layers: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> nn.Sequential:
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

    def _require_observation_batch(self, obs: Tensor) -> Tensor:
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2D (batch, observation), got shape {tuple(obs.shape)}")
        if obs.shape[1] != self.observation_dim:
            raise ValueError(
                f"obs feature dimension mismatch: expected {self.observation_dim}, got {obs.shape[1]}"
            )
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
            raise ValueError(
                f"hidden_state must be 2D (batch, hidden_size), got shape {tuple(hidden_state.shape)}"
            )
        if hidden_state.shape[0] != batch_size:
            raise ValueError(
                f"hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}"
            )
        if hidden_state.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_state feature mismatch: expected {self.hidden_size}, got {hidden_state.shape[1]}"
            )
        return hidden_state.to(device=like.device, dtype=like.dtype)

    def _prepare_seat_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        if hidden_state is None:
            return self.initial_seat_hidden(batch_size, device=like.device, dtype=like.dtype)
        if hidden_state.ndim != 3:
            raise ValueError(
                f"seat_hidden_state must be 3D (batch, seat, hidden_size), got shape {tuple(hidden_state.shape)}"
            )
        if hidden_state.shape[0] != batch_size:
            raise ValueError(
                f"seat_hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}"
            )
        if hidden_state.shape[1] != SEAT_COUNT:
            raise ValueError(
                f"seat_hidden_state seat mismatch: expected {SEAT_COUNT}, got {hidden_state.shape[1]}"
            )
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
                    raise ValueError(
                        f"acting_seat batch mismatch: expected {batch_size}, got {acting_seat.shape[0]}"
                    )
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
        batch_index = torch.arange(seat_hidden_state.shape[0], device=seat_hidden_state.device)
        next_seat_hidden[batch_index, acting_seat] = next_acting_hidden
        return next_seat_hidden


__all__ = ["GLOBAL_ACTION_SPACE_SIZE", "SEAT_COUNT", "ModelConfig", "PolicyValueModel"]
