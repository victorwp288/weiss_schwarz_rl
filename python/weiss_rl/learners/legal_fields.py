"""Learner legal-action and core batch field validators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch

BatchValueGetter = Callable[[Any, str], Any]


def require_obs(value: Any, *, reference: Tensor) -> Tensor:
    """Convert and validate a time-major observation tensor."""
    if value is None:
        raise ValueError("batch field is required")
    tensor = torch.as_tensor(value, device=reference.device).to(dtype=reference.dtype)
    if tensor.ndim != 3:
        raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(tensor.shape)}")
    return tensor


def require_actions(value: Any, *, expected_shape: torch.Size, reference: Tensor) -> Tensor:
    """Convert and validate a time-major action tensor."""
    if value is None:
        raise ValueError("batch field is required")
    tensor = torch.as_tensor(value, device=reference.device).to(dtype=torch.long)
    if tensor.shape != expected_shape:
        raise ValueError(f"actions must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
    return tensor


def require_legal_mask(value: Any, *, expected_shape: torch.Size, reference: Tensor) -> Tensor:
    """Convert and validate a dense time-major legal-action mask."""
    if value is None:
        raise ValueError("batch field is required")
    tensor = torch.as_tensor(value, device=reference.device).to(dtype=torch.bool)
    if tensor.ndim != 3 or tensor.shape[:2] != expected_shape:
        expected = (int(expected_shape[0]), int(expected_shape[1]), "action")
        raise ValueError(f"legal_mask must have shape {expected}, got {tuple(tensor.shape)}")
    return tensor


def has_legal_actions(batch: Any, *, batch_value: BatchValueGetter) -> bool:
    """Return whether a learner batch contains any supported legality representation."""
    if batch_value(batch, "legal_actions") is not None:
        return True
    if batch_value(batch, "legal_mask") is not None:
        return True
    return batch_value(batch, "legal_ids") is not None and batch_value(batch, "legal_offsets") is not None


def resolve_legal_mask(
    batch: Any,
    *,
    expected_shape: torch.Size,
    action_dim: int,
    reference: Tensor,
    batch_value: BatchValueGetter,
) -> Tensor:
    """Resolve dense legal masks from dense, packed, or LegalActionBatch inputs."""
    legal_actions = batch_value(batch, "legal_actions")
    if isinstance(legal_actions, LegalActionBatch):
        mask = legal_actions.to_mask(
            expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
            action_space=action_dim,
        )
        return torch.as_tensor(mask, dtype=torch.bool, device=reference.device)

    legal_mask = batch_value(batch, "legal_mask")
    if legal_mask is not None:
        return require_legal_mask(legal_mask, expected_shape=expected_shape, reference=reference)

    legal_ids = batch_value(batch, "legal_ids")
    legal_offsets = batch_value(batch, "legal_offsets")
    if legal_ids is None or legal_offsets is None:
        raise ValueError("batch must include either legal_actions, legal_mask, or legal_ids/legal_offsets")
    mask = LegalActionBatch.from_packed(legal_ids, legal_offsets).to_mask(
        expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
        action_space=action_dim,
    )
    return torch.as_tensor(mask, dtype=torch.bool, device=reference.device)


def resolve_packed_legal_actions_with_meta(
    batch: Any,
    *,
    expected_shape: torch.Size,
    reference: Tensor,
    batch_value: BatchValueGetter,
    supports_legal_candidate_scoring: bool,
) -> tuple[Tensor, Tensor, Tensor | None] | None:
    """Resolve packed legal-action ids, offsets, and optional metadata."""
    legal_actions = batch_value(batch, "legal_actions")
    if (
        isinstance(legal_actions, LegalActionBatch)
        and legal_actions.ids is not None
        and legal_actions.offsets is not None
    ):
        ids = torch.as_tensor(legal_actions.ids, device=reference.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=reference.device, dtype=torch.long)
        expected_rows = int(expected_shape[0] * expected_shape[1])
        if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
            raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
        meta = (
            None
            if legal_actions.meta is None
            else torch.as_tensor(legal_actions.meta, device=reference.device, dtype=torch.long)
        )
        if supports_legal_candidate_scoring and meta is None:
            raise ValueError("structured learner updates require packed legal action metadata")
        return ids, offsets, meta

    legal_ids = batch_value(batch, "legal_ids")
    legal_offsets = batch_value(batch, "legal_offsets")
    if legal_ids is None or legal_offsets is None:
        return None
    ids = torch.as_tensor(legal_ids, device=reference.device, dtype=torch.long)
    offsets = torch.as_tensor(legal_offsets, device=reference.device, dtype=torch.long)
    expected_rows = int(expected_shape[0] * expected_shape[1])
    if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
        raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
    legal_action_meta = batch_value(batch, "legal_action_meta")
    meta = (
        None
        if legal_action_meta is None
        else torch.as_tensor(legal_action_meta, device=reference.device, dtype=torch.long)
    )
    if supports_legal_candidate_scoring and meta is None:
        raise ValueError("structured learner updates require packed legal action metadata")
    return ids, offsets, meta
