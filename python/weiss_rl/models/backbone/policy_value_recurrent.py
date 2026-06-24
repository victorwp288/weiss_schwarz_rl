"""Recurrent hidden-state mechanics for policy/value models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.backbone.state import (
    prepare_acting_seat,
    prepare_hidden_state,
    prepare_seat_hidden_state,
    select_acting_hidden,
    write_acting_hidden,
)

SEAT_COUNT = 2


class PolicyValueRecurrentMixin:
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
        opponent_policy_ids: Sequence[object] | None = None,
        opponent_context_indices: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        hidden = torch.zeros(batch_size, SEAT_COUNT, self.hidden_size, device=hidden_device, dtype=hidden_dtype)
        context = self._opponent_context_hidden(
            batch_size=batch_size,
            device=hidden_device,
            dtype=hidden_dtype,
            opponent_policy_ids=opponent_policy_ids,
            opponent_context_indices=opponent_context_indices,
        )
        if context is not None:
            hidden = hidden + context.unsqueeze(1)
        return hidden

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

    def _hidden_tensor_device_dtype(
        self,
        *,
        batch_size: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        reference = self._reference_parameter()
        hidden_device: torch.device = reference.device if device is None else device
        hidden_dtype: torch.dtype = reference.dtype if dtype is None else dtype
        return hidden_device, hidden_dtype

    def _prepare_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        return prepare_hidden_state(
            hidden_state,
            batch_size=batch_size,
            like=like,
            hidden_size=self.hidden_size,
            initial_hidden=lambda current_batch_size: self.initial_hidden(
                current_batch_size,
                device=like.device,
                dtype=like.dtype,
            ),
        )

    def _prepare_seat_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        return prepare_seat_hidden_state(
            hidden_state,
            batch_size=batch_size,
            like=like,
            hidden_size=self.hidden_size,
            seat_count=SEAT_COUNT,
            initial_seat_hidden=lambda current_batch_size: self.initial_seat_hidden(
                current_batch_size,
                device=like.device,
                dtype=like.dtype,
            ),
        )

    def _prepare_acting_seat(self, acting_seat: int | Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        return prepare_acting_seat(acting_seat, batch_size=batch_size, device=device)

    def _select_acting_hidden(self, seat_hidden_state: Tensor, acting_seat: Tensor) -> Tensor:
        return select_acting_hidden(seat_hidden_state, acting_seat, hidden_size=self.hidden_size)

    def _write_acting_hidden(
        self,
        seat_hidden_state: Tensor,
        acting_seat: Tensor,
        next_acting_hidden: Tensor,
    ) -> Tensor:
        return write_acting_hidden(seat_hidden_state, acting_seat, next_acting_hidden)

    def _reference_parameter(self: Any) -> Tensor:
        try:
            return next(self.parameters())
        except StopIteration as exc:
            raise RuntimeError("Model has no parameters to use as a reference tensor") from exc


__all__ = [
    "PolicyValueRecurrentMixin",
    "SEAT_COUNT",
]
