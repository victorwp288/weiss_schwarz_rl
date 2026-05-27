"""Batch field conversion and validation helpers for learners."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def tensor_on_device(value: Any, *, reference: Tensor, dtype: torch.dtype) -> Tensor:
    """Convert a required batch value to the reference tensor device and dtype."""
    if value is None:
        raise ValueError("batch field is required")
    tensor = torch.as_tensor(value, device=reference.device)
    return tensor.to(dtype=dtype)


def float_target(value: Any, *, expected_shape: torch.Size, like: Tensor, reference: Tensor) -> Tensor:
    """Convert a target tensor and verify its exact shape."""
    tensor = tensor_on_device(value, reference=reference, dtype=like.dtype)
    if tensor.shape != expected_shape:
        raise ValueError(f"target must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor


def optional_batch_seat_field(
    value: Any,
    *,
    field_name: str,
    expected_batch_size: int,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional batch-length seat field with values constrained to 0/1."""
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=reference.device)
    if tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{field_name} must be integer-valued")
    tensor = tensor.to(dtype=torch.long)
    if tensor.ndim != 1 or tensor.shape[0] != expected_batch_size:
        raise ValueError(f"{field_name} must have shape ({expected_batch_size},), got {tuple(tensor.shape)}")
    if bool(((tensor != 0) & (tensor != 1)).any().item()):
        raise ValueError(f"{field_name} values must be 0 or 1")
    return tensor


def prepare_legacy_hidden_state(value: Any, *, batch_size: int, like: Tensor, reference: Tensor) -> Tensor | None:
    """Validate legacy two-dimensional hidden state batches."""
    if value is None:
        return None
    tensor = tensor_on_device(value, reference=reference, dtype=like.dtype)
    if tensor.ndim != 2:
        raise ValueError(
            "initial_hidden_state must be 2D (batch, hidden_size) when to_play_seat/actor is absent, "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != batch_size:
        raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    return tensor


def prepare_seat_hidden_state(value: Any, *, batch_size: int, like: Tensor, reference: Tensor) -> Tensor | None:
    """Validate seat-aware three-dimensional hidden state batches."""
    if value is None:
        return None
    tensor = tensor_on_device(value, reference=reference, dtype=like.dtype)
    if tensor.ndim != 3:
        raise ValueError(
            "initial_hidden_state must be 3D (batch, seat, hidden_size) when to_play_seat/actor is present, "
            f"got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != batch_size:
        raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    if tensor.shape[1] != 2:
        raise ValueError(f"initial_hidden_state seat mismatch: expected 2, got {tensor.shape[1]}")
    return tensor


def optional_time_major_seat_field(
    value: Any,
    *,
    field_name: str,
    expected_shape: torch.Size,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional time-major seat field with values constrained to 0/1."""
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=reference.device)
    if tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{field_name} must be integer-valued")
    tensor = tensor.to(dtype=torch.long)
    if tensor.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    if bool(((tensor != 0) & (tensor != 1)).any().item()):
        raise ValueError(f"{field_name} values must be 0 or 1")
    return tensor


def prepare_acting_seat_batch(
    to_play_seat: Any,
    *,
    actor: Any,
    expected_shape: torch.Size,
    reference: Tensor,
) -> Tensor | None:
    """Resolve equivalent ``to_play_seat`` and ``actor`` batch fields."""
    seat_tensor = optional_time_major_seat_field(
        to_play_seat,
        field_name="to_play_seat",
        expected_shape=expected_shape,
        reference=reference,
    )
    actor_tensor = optional_time_major_seat_field(
        actor,
        field_name="actor",
        expected_shape=expected_shape,
        reference=reference,
    )

    if seat_tensor is None:
        return actor_tensor
    if actor_tensor is None:
        return seat_tensor
    if not torch.equal(seat_tensor, actor_tensor):
        raise ValueError("actor must match to_play_seat when both are provided")
    return seat_tensor


def optional_time_major_loss_mask(
    value: Any,
    *,
    expected_shape: torch.Size,
    like: Tensor,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional policy-train mask and clamp it into [0, 1]."""
    if value is None:
        return None
    tensor = tensor_on_device(value, reference=reference, dtype=like.dtype)
    if tensor.shape != expected_shape:
        raise ValueError(f"policy_train_mask must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor.clamp(min=0.0, max=1.0)


def optional_time_major_float_field(
    value: Any,
    *,
    field_name: str,
    expected_shape: torch.Size,
    like: Tensor,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional float-valued time-major field without clamping."""
    if value is None:
        return None
    tensor = tensor_on_device(value, reference=reference, dtype=like.dtype)
    if tensor.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor


def optional_time_major_index_field(
    value: Any,
    *,
    field_name: str,
    expected_shape: torch.Size,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional integer-valued time-major index field."""
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=reference.device)
    if tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{field_name} must be integer-valued")
    tensor = tensor.to(dtype=torch.long)
    if tensor.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor


def optional_time_major_bool_field(
    value: Any,
    *,
    field_name: str,
    expected_shape: torch.Size,
    reference: Tensor,
) -> Tensor | None:
    """Convert an optional time-major field to bool and verify shape."""
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=reference.device, dtype=torch.bool)
    if tensor.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor
