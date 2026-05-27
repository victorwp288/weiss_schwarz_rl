"""Structured model helpers for gathering stage-slot features."""

from __future__ import annotations

import torch
from torch import Tensor


def gather_stage_features_for_rows(
    slot_contexts: Tensor,
    slot_numeric: Tensor,
    row_indices: Tensor,
    slot_indices: Tensor,
    *,
    stage_slot_count: int,
) -> tuple[Tensor, Tensor]:
    """Gather per-row stage features and zero invalid slot references."""

    valid = (slot_indices >= 0) & (slot_indices < stage_slot_count)
    if not torch.any(valid):
        return (
            slot_contexts.new_zeros((slot_indices.shape[0], slot_contexts.shape[-1])),
            slot_numeric.new_zeros((slot_indices.shape[0], slot_numeric.shape[-1])),
        )
    safe_rows = torch.where(valid, row_indices, torch.zeros_like(row_indices)).to(dtype=torch.long)
    safe_slots = torch.where(valid, slot_indices, torch.zeros_like(slot_indices)).to(dtype=torch.long)
    flat_indices = safe_rows * stage_slot_count + safe_slots
    gathered_context = slot_contexts.reshape(-1, slot_contexts.shape[-1]).index_select(0, flat_indices)
    gathered_numeric = slot_numeric.reshape(-1, slot_numeric.shape[-1]).index_select(0, flat_indices)
    return (
        gathered_context * valid.unsqueeze(1).to(dtype=slot_contexts.dtype),
        gathered_numeric * valid.unsqueeze(1).to(dtype=slot_numeric.dtype),
    )


def gather_stage_features(
    slot_contexts: Tensor,
    slot_numeric: Tensor,
    slot_indices: Tensor,
    *,
    stage_slot_count: int,
) -> tuple[Tensor, Tensor]:
    """Gather stage features from either batched slots or a shared slot table."""

    if slot_contexts.ndim == 3:
        valid = (slot_indices >= 0) & (slot_indices < stage_slot_count)
        safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
        context_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_contexts.shape[-1])
        numeric_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_numeric.shape[-1])
        gathered_context = torch.gather(slot_contexts, 1, context_index).squeeze(1)
        gathered_numeric = torch.gather(slot_numeric, 1, numeric_index).squeeze(1)
        return (
            gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
            gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
        )
    valid = (slot_indices >= 0) & (slot_indices < stage_slot_count)
    safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
    gathered_context = slot_contexts.index_select(0, safe_indices.to(dtype=torch.long))
    gathered_numeric = slot_numeric.index_select(0, safe_indices.to(dtype=torch.long))
    return (
        gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
        gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
    )


def slot_component(stage_values: Tensor, offset: int) -> Tensor:
    """Return a stage component slice or a zero plane when the component is absent."""

    if offset >= stage_values.shape[-1]:
        return torch.zeros(stage_values.shape[:2], device=stage_values.device, dtype=stage_values.dtype)
    return stage_values[..., offset]
