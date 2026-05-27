"""Small tensor helpers shared by learner loss code."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def segment_max(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    """Return the maximum value for each segment key.

    Segments with no values keep ``-inf``.  This mirrors the learner's packed
    legal-action reductions where some batch rows can have no selected entries.
    """
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    if keys.numel() == 0:
        return out
    out.scatter_reduce_(0, keys.to(dtype=torch.long), values, reduce="amax", include_self=True)
    return out


def segment_logsumexp(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    """Compute logsumexp grouped by segment key, preserving empty segments as ``-inf``."""
    num_segments = int(num_segments)
    max_per = segment_max(values, keys, num_segments)
    if keys.numel() == 0:
        return max_per
    long_keys = keys.to(dtype=torch.long)
    gathered_max = max_per.index_select(0, long_keys)
    shifted = torch.exp(values - gathered_max)
    sumexp = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, long_keys, shifted)
    valid = torch.isfinite(max_per) & (sumexp > 0)
    out = torch.full((num_segments,), -torch.inf, dtype=values.dtype, device=values.device)
    out[valid] = torch.log(sumexp[valid]) + max_per[valid]
    return out


def segment_group_sum(
    values: Tensor,
    row_indices: Tensor,
    group_ids: Tensor,
    *,
    row_count: int,
    group_count: int,
) -> Tensor:
    """Sum values into a dense ``[row_count, group_count]`` table."""
    row_count = int(row_count)
    group_count = int(group_count)
    if group_count <= 0:
        return torch.zeros((row_count, 0), dtype=values.dtype, device=values.device)
    out = torch.zeros((row_count * group_count,), dtype=values.dtype, device=values.device)
    if values.numel() == 0:
        return out.view(row_count, group_count)
    valid = (group_ids >= 0) & (group_ids < group_count)
    if not bool(valid.any().item()):
        return out.view(row_count, group_count)
    flat_keys = row_indices[valid].to(dtype=torch.long) * group_count + group_ids[valid].to(dtype=torch.long)
    out.scatter_add_(0, flat_keys, values[valid])
    return out.view(row_count, group_count)


def weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    """Average values with the learner's historical clamped denominator."""
    denominator = torch.clamp(weight.sum(), min=1.0)
    return (value * weight).sum() / denominator


def nonfinite_indices(values: Tensor | np.ndarray) -> np.ndarray:
    """Return index coordinates for NaN or infinite entries."""
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return np.argwhere(~np.isfinite(array)).astype(np.int64, copy=False)
