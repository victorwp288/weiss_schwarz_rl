"""V-trace bootstrap helpers for IMPALA learner updates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.batch_fields import optional_batch_seat_field, tensor_on_device

BatchValueGetter = Callable[[Any, str], Any]
ReferenceParameter = Callable[[], Tensor]


def has_raw_vtrace_inputs(batch: Any, *, batch_value: BatchValueGetter) -> bool:
    """Return whether a batch has enough raw fields to compute V-trace targets."""

    if any(batch_value(batch, key) is None for key in ("rewards", "discounts", "behavior_logp")):
        return False
    if batch_value(batch, "bootstrap_value") is not None:
        return True
    return all(
        batch_value(batch, key) is not None for key in ("bootstrap_obs", "bootstrap_actor", "final_hidden_state")
    )


def resolve_vtrace_bootstrap_value(
    batch: Any,
    *,
    batch_size: int,
    like: Tensor,
    model: Any,
    compiled_model: Any,
    reference_parameter: ReferenceParameter,
    batch_value: BatchValueGetter,
) -> Tensor:
    """Resolve the bootstrap value from current model fields or stored batch values."""

    current_bootstrap = current_model_bootstrap_value(
        batch,
        batch_size=batch_size,
        like=like,
        model=model,
        compiled_model=compiled_model,
        reference_parameter=reference_parameter,
        batch_value=batch_value,
    )
    if current_bootstrap is not None:
        return current_bootstrap
    reference = reference_parameter()
    bootstrap_value = tensor_on_device(
        batch_value(batch, "bootstrap_value"), reference=reference, dtype=reference.dtype
    )
    if bootstrap_value.ndim != 1 or bootstrap_value.shape[0] != batch_size:
        raise ValueError(f"bootstrap_value must have shape ({batch_size},), got {tuple(bootstrap_value.shape)}")
    return bootstrap_value


def current_model_bootstrap_value(
    batch: Any,
    *,
    batch_size: int,
    like: Tensor,
    model: Any,
    compiled_model: Any,
    reference_parameter: ReferenceParameter,
    batch_value: BatchValueGetter,
) -> Tensor | None:
    """Evaluate current model bootstrap values for valid final actor rows."""

    bootstrap_obs_value = batch_value(batch, "bootstrap_obs")
    bootstrap_actor_value = batch_value(batch, "bootstrap_actor")
    final_hidden_value = batch_value(batch, "final_hidden_state")
    if bootstrap_obs_value is None or bootstrap_actor_value is None or final_hidden_value is None:
        return None
    if model is None:
        return None
    forward_model = compiled_model if compiled_model is not None else model
    if not hasattr(forward_model, "value_seat_aware") and not hasattr(forward_model, "forward_seat_aware"):
        return None
    reference = reference_parameter()
    bootstrap_obs = tensor_on_device(bootstrap_obs_value, reference=reference, dtype=like.dtype)
    if bootstrap_obs.ndim != 2 or bootstrap_obs.shape[0] != batch_size:
        raise ValueError(f"bootstrap_obs must have shape ({batch_size}, observation), got {tuple(bootstrap_obs.shape)}")
    bootstrap_actor = optional_batch_seat_field(
        bootstrap_actor_value,
        field_name="bootstrap_actor",
        expected_batch_size=batch_size,
        reference=reference,
    )
    if bootstrap_actor is None:
        return None
    final_hidden_state = tensor_on_device(final_hidden_value, reference=reference, dtype=like.dtype)
    if final_hidden_state.ndim != 3:
        return None
    if final_hidden_state.shape[0] != batch_size:
        raise ValueError(f"final_hidden_state batch mismatch: expected {batch_size}, got {final_hidden_state.shape[0]}")
    if final_hidden_state.shape[1] != 2:
        raise ValueError(f"final_hidden_state seat mismatch: expected 2, got {final_hidden_state.shape[1]}")
    valid_rows = ((bootstrap_actor == 0) | (bootstrap_actor == 1)).to(dtype=torch.bool)
    bootstrap_value = torch.zeros((batch_size,), dtype=like.dtype, device=like.device)
    if not bool(valid_rows.any().item()):
        return bootstrap_value
    with torch.no_grad():
        value_seat_aware = getattr(forward_model, "value_seat_aware", None)
        if callable(value_seat_aware):
            value_tensor = value_seat_aware(
                bootstrap_obs[valid_rows],
                bootstrap_actor[valid_rows],
                final_hidden_state[valid_rows],
            )
        else:
            _logits_tensor, value_tensor, _next_hidden = forward_model.forward_seat_aware(
                bootstrap_obs[valid_rows],
                bootstrap_actor[valid_rows],
                final_hidden_state[valid_rows],
            )
    bootstrap_value[valid_rows] = torch.as_tensor(
        value_tensor,
        device=bootstrap_value.device,
        dtype=bootstrap_value.dtype,
    )
    return bootstrap_value
