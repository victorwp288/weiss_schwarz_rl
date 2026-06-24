from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def require_observation_batch(obs: Tensor, *, observation_dim: int, dtype: torch.dtype) -> Tensor:
    if obs.ndim != 2:
        raise ValueError(f"obs must be 2D (batch, observation), got shape {tuple(obs.shape)}")
    if obs.shape[1] != observation_dim:
        raise ValueError(f"obs feature dimension mismatch: expected {observation_dim}, got {obs.shape[1]}")
    return obs.to(dtype=dtype)


def prepare_hidden_state(
    hidden_state: Tensor | None,
    *,
    batch_size: int,
    like: Tensor,
    hidden_size: int,
    initial_hidden: Callable[[int], Tensor],
) -> Tensor:
    if hidden_state is None:
        return initial_hidden(batch_size)
    if hidden_state.ndim != 2:
        raise ValueError(f"hidden_state must be 2D (batch, hidden_size), got shape {tuple(hidden_state.shape)}")
    if hidden_state.shape[0] != batch_size:
        raise ValueError(f"hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
    if hidden_state.shape[1] != hidden_size:
        raise ValueError(f"hidden_state feature mismatch: expected {hidden_size}, got {hidden_state.shape[1]}")
    return hidden_state.to(device=like.device, dtype=like.dtype)


def prepare_seat_hidden_state(
    hidden_state: Tensor | None,
    *,
    batch_size: int,
    like: Tensor,
    hidden_size: int,
    seat_count: int,
    initial_seat_hidden: Callable[[int], Tensor],
) -> Tensor:
    if hidden_state is None:
        return initial_seat_hidden(batch_size)
    if hidden_state.ndim != 3:
        raise ValueError(
            f"seat_hidden_state must be 3D (batch, seat, hidden_size), got shape {tuple(hidden_state.shape)}"
        )
    if hidden_state.shape[0] != batch_size:
        raise ValueError(f"seat_hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
    if hidden_state.shape[1] != seat_count:
        raise ValueError(f"seat_hidden_state seat mismatch: expected {seat_count}, got {hidden_state.shape[1]}")
    if hidden_state.shape[2] != hidden_size:
        raise ValueError(f"seat_hidden_state feature mismatch: expected {hidden_size}, got {hidden_state.shape[2]}")
    return hidden_state.to(device=like.device, dtype=like.dtype)


def prepare_acting_seat(acting_seat: int | Tensor, *, batch_size: int, device: torch.device) -> Tensor:
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


def select_acting_hidden(seat_hidden_state: Tensor, acting_seat: Tensor, *, hidden_size: int) -> Tensor:
    acting_index = acting_seat.view(-1, 1, 1).expand(-1, 1, hidden_size)
    return torch.gather(seat_hidden_state, dim=1, index=acting_index).squeeze(1)


def write_acting_hidden(
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
