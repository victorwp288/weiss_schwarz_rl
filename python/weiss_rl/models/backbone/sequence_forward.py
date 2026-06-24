"""Sequence unroll helpers for recurrent policy/value models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch


def forward_sequence_seat_aware_dense(
    model: Any,
    obs: Tensor,
    acting_seat: Tensor,
    seat_hidden_state: Tensor | None = None,
    *,
    legal_actions: LegalActionBatch | None = None,
    reset_before_step: Tensor | None = None,
    opponent_context_index: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if legal_actions is not None:
        raise ValueError("forward_sequence_seat_aware with legal_actions is only supported on structured models")
    if obs.ndim != 3:
        raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
    if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
        raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")

    batch_size = int(obs.shape[1])
    seat_hidden = model._prepare_seat_hidden_state(seat_hidden_state, batch_size=batch_size, like=obs[0])
    reset_mask = _optional_sequence_bool_mask(reset_before_step, obs=obs, name="reset_before_step")
    context_index = _optional_sequence_long_mask(opponent_context_index, obs=obs, name="opponent_context_index")

    logits_steps: list[Tensor] = []
    value_steps: list[Tensor] = []
    for step_index, (step_obs, step_seat) in enumerate(zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)):
        if reset_mask is not None:
            seat_hidden = _reset_finished_rows(
                model,
                seat_hidden,
                reset_mask[step_index],
                context_index=None if context_index is None else context_index[step_index],
            )
        step_context = None if context_index is None else context_index[step_index]
        step_logits, step_value, seat_hidden = model.forward_seat_aware(
            step_obs,
            step_seat,
            seat_hidden,
            opponent_context_index=step_context,
        )
        logits_steps.append(step_logits)
        value_steps.append(step_value)
    return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0), seat_hidden


def _optional_sequence_bool_mask(mask: Tensor | None, *, obs: Tensor, name: str) -> Tensor | None:
    if mask is None:
        return None
    mask_tensor = torch.as_tensor(mask, device=obs.device, dtype=torch.bool)
    if mask_tensor.ndim != 2 or mask_tensor.shape != obs.shape[:2]:
        raise ValueError(f"{name} must be 2D (time, batch) with the same leading dimensions as obs")
    return mask_tensor


def _optional_sequence_long_mask(mask: Tensor | None, *, obs: Tensor, name: str) -> Tensor | None:
    if mask is None:
        return None
    mask_tensor = torch.as_tensor(mask, device=obs.device, dtype=torch.long)
    if mask_tensor.ndim != 2 or mask_tensor.shape != obs.shape[:2]:
        raise ValueError(f"{name} must be 2D (time, batch) with the same leading dimensions as obs")
    return mask_tensor


def _reset_finished_rows(
    model: Any,
    seat_hidden: Tensor,
    step_reset: Tensor,
    *,
    context_index: Tensor | None,
) -> Tensor:
    if not bool(step_reset.any().item()):
        return seat_hidden

    reset_rows = torch.nonzero(step_reset, as_tuple=False).squeeze(1)
    next_seat_hidden = seat_hidden.clone()
    next_seat_hidden.index_copy_(
        0,
        reset_rows,
        model.initial_seat_hidden(
            int(reset_rows.numel()),
            device=seat_hidden.device,
            dtype=seat_hidden.dtype,
            opponent_context_indices=None if context_index is None else context_index.index_select(0, reset_rows),
        ),
    )
    return next_seat_hidden


__all__ = ["forward_sequence_seat_aware_dense"]
