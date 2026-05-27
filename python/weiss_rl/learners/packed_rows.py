"""Packed legal-action row slicing helpers for learner update paths."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor


def packed_legal_action_view(packed_legal: tuple[Tensor, Tensor, Tensor | None]) -> Any:
    """Return the lightweight legal-action view expected by model scorers."""

    ids, offsets, meta = packed_legal
    return SimpleNamespace(ids=ids, offsets=offsets, meta=meta)


def slice_packed_legal_rows_with_meta(
    packed_legal: tuple[Tensor, Tensor, Tensor | None],
    row_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Slice packed legal ids, offsets, and optional metadata by row indices."""

    ids, offsets, meta = packed_legal
    row_indices = row_indices.to(device=offsets.device, dtype=torch.long)
    subset_offsets = offsets.new_zeros((int(row_indices.shape[0]) + 1,))
    if row_indices.numel() == 0:
        empty_meta = None if meta is None else meta.new_zeros((0, meta.shape[1]))
        return ids.new_zeros((0,)), subset_offsets, empty_meta
    widths = offsets[1:] - offsets[:-1]
    selected_widths = widths.index_select(0, row_indices).to(dtype=torch.long)
    subset_offsets[1:] = torch.cumsum(selected_widths.to(dtype=offsets.dtype), dim=0)
    total = int(subset_offsets[-1].item())
    if total == 0:
        empty_meta = None if meta is None else meta.new_zeros((0, meta.shape[1]))
        return ids.new_zeros((0,)), subset_offsets, empty_meta
    flat_positions = packed_candidate_positions_for_rows(offsets, row_indices)
    subset_ids = ids.index_select(0, flat_positions)
    subset_meta = None if meta is None else meta.index_select(0, flat_positions)
    return subset_ids, subset_offsets, subset_meta


def packed_candidate_positions_for_rows(offsets: Tensor, row_indices: Tensor) -> Tensor:
    """Return flat packed-candidate positions for the selected rows."""

    row_indices = row_indices.to(device=offsets.device, dtype=torch.long)
    if row_indices.numel() == 0:
        return row_indices.new_zeros((0,))
    widths = offsets[1:] - offsets[:-1]
    selected_widths = widths.index_select(0, row_indices).to(dtype=torch.long)
    total = int(selected_widths.sum().item())
    if total == 0:
        return row_indices.new_zeros((0,))
    row_starts = offsets.index_select(0, row_indices).to(dtype=torch.long)
    local_offsets = offsets.new_zeros((int(row_indices.shape[0]) + 1,))
    local_offsets[1:] = torch.cumsum(selected_widths.to(dtype=offsets.dtype), dim=0)
    local_starts = local_offsets[:-1].to(dtype=torch.long)
    repeated_row_starts = torch.repeat_interleave(row_starts, selected_widths)
    repeated_local_starts = torch.repeat_interleave(local_starts, selected_widths)
    return repeated_row_starts + (torch.arange(total, device=offsets.device, dtype=torch.long) - repeated_local_starts)


def scatter_packed_candidate_values(
    packed_legal: tuple[Tensor, Tensor, Tensor | None],
    row_indices: Tensor,
    subset_values: Tensor,
    *,
    fill_value: float = 0.0,
) -> Tensor:
    """Scatter selected packed candidate values back into the full packed candidate vector."""

    ids, offsets, _meta = packed_legal
    full = subset_values.new_full((int(ids.shape[0]),), fill_value)
    if row_indices.numel() == 0 or subset_values.numel() == 0:
        return full
    flat_positions = packed_candidate_positions_for_rows(offsets, row_indices)
    full.index_copy_(0, flat_positions.to(device=full.device), subset_values.reshape(-1))
    return full


def subset_observation_context_rows(
    observation_context: Mapping[str, Tensor],
    row_indices: Tensor,
    *,
    row_count: int,
) -> dict[str, Tensor]:
    """Subset row-major tensor entries in an observation context mapping."""

    subset: dict[str, Tensor] = {}
    for key, value in observation_context.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == row_count:
            subset[key] = value.index_select(0, row_indices)
        else:
            subset[key] = value
    return subset
