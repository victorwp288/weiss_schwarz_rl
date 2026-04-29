"""Packed and masked sampling helpers for structured action scoring."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def negative_logits_fill_value(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).min)
    return -1.0e9


def packed_row_indices(offsets: Tensor) -> Tensor:
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(int(lengths.shape[0]), device=offsets.device, dtype=torch.long),
        lengths.to(dtype=torch.long),
    )


def packed_row_log_z(scores: Tensor, offsets: Tensor) -> Tensor:
    row_count = int(offsets.shape[0] - 1)
    if row_count < 0:
        raise ValueError("packed offsets must contain at least one row boundary")
    row_log_z = torch.full((row_count,), -torch.inf, device=scores.device, dtype=scores.dtype)
    if scores.numel() == 0 or row_count == 0:
        return row_log_z
    lengths = offsets[1:] - offsets[:-1]
    non_empty_rows = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return row_log_z
    non_empty_lengths = lengths[non_empty_rows].to(dtype=torch.long)
    segment_max = torch.segment_reduce(scores, reduce="max", lengths=non_empty_lengths)
    repeated_max = torch.repeat_interleave(segment_max, non_empty_lengths)
    shifted = scores - repeated_max
    exp_shifted = torch.exp(shifted)
    segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_lengths)
    row_log_z[non_empty_rows] = torch.log(segment_sum) + segment_max
    return row_log_z


def packed_local_cdf(probabilities: Tensor, offsets: Tensor) -> Tensor:
    if probabilities.numel() == 0:
        return probabilities
    row_count = int(offsets.shape[0] - 1)
    row_indices = packed_row_indices(offsets)
    cumulative = torch.cumsum(probabilities, dim=0)
    base = torch.zeros((row_count,), dtype=probabilities.dtype, device=probabilities.device)
    if row_count > 1:
        starts = offsets[1:-1].to(dtype=torch.long)
        base[1:] = cumulative.index_select(0, starts - 1)
    return cumulative - base.index_select(0, row_indices)


def uniform_from_seeds(sample_seeds: Tensor, *, dtype: torch.dtype) -> Tensor:
    seed_float = sample_seeds.to(dtype=torch.float64)
    hashed = torch.sin(seed_float * 12.9898 + 78.233) * 43758.5453123
    uniform = torch.frac(hashed).to(dtype=dtype)
    eps = torch.finfo(dtype).eps
    return torch.clamp(uniform, min=eps, max=1.0 - eps)


def derived_sample_seeds(sample_seeds: Tensor, *, salt: int) -> Tensor:
    mixed = sample_seeds.to(dtype=torch.long)
    return mixed ^ torch.full_like(mixed, int(salt), dtype=torch.long)


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("masked_log_softmax requires logits and mask with matching shapes")
    negative_fill = torch.full_like(logits, negative_logits_fill_value(logits.dtype))
    masked_logits = torch.where(mask, logits, negative_fill)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return torch.where(mask, log_probs, negative_fill)


def masked_entropy_from_log_probs(log_probs: Tensor, mask: Tensor) -> Tensor:
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


def sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
    uniform_fn: Callable[..., Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    uniform_fn = uniform_from_seeds if uniform_fn is None else uniform_fn
    if log_probs.ndim != 2 or mask.ndim != 2 or log_probs.shape != mask.shape:
        raise ValueError("sampled masked log_probs requires 2D tensors with matching shape")
    row_count = int(log_probs.shape[0])
    if sample_seeds.ndim != 1 or int(sample_seeds.shape[0]) != row_count:
        raise ValueError(f"sample_seeds must have shape ({row_count},)")
    actions = torch.full((row_count,), int(default_index), device=log_probs.device, dtype=torch.long)
    selected_logp = torch.zeros((row_count,), device=log_probs.device, dtype=log_probs.dtype)
    if row_count == 0:
        return actions, selected_logp
    row_has_candidates = mask.any(dim=1)
    non_empty_rows = torch.nonzero(row_has_candidates, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return actions, selected_logp
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    cdf = torch.cumsum(probs, dim=1)
    thresholds = uniform_fn(
        sample_seeds.index_select(0, non_empty_rows).to(device=log_probs.device, dtype=torch.long),
        dtype=log_probs.dtype,
    ).unsqueeze(1)
    cdf_rows = cdf.index_select(0, non_empty_rows)
    chosen = cdf_rows >= thresholds
    chosen_indices = chosen.to(dtype=torch.int64).argmax(dim=1)
    fallback_indices = mask.index_select(0, non_empty_rows).to(dtype=torch.int64).argmax(dim=1)
    chosen_indices = torch.where(chosen.any(dim=1), chosen_indices, fallback_indices)
    actions[non_empty_rows] = chosen_indices
    selected_logp[non_empty_rows] = (
        log_probs.index_select(0, non_empty_rows)
        .gather(
            1,
            chosen_indices.unsqueeze(1),
        )
        .squeeze(1)
    )
    return actions, selected_logp


def sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
    uniform_fn: Callable[..., Tensor] | None = None,
    packed_local_cdf_fn: Callable[..., Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    uniform_fn = uniform_from_seeds if uniform_fn is None else uniform_fn
    packed_local_cdf_fn = packed_local_cdf if packed_local_cdf_fn is None else packed_local_cdf_fn
    if packed_scores.ndim != 1:
        raise ValueError("packed_scores must be 1D")
    if packed_ids.ndim != 1 or packed_offsets.ndim != 1:
        raise ValueError("packed ids and offsets must be 1D")
    row_count = int(packed_offsets.shape[0] - 1)
    if sample_seeds.ndim != 1 or int(sample_seeds.shape[0]) != row_count:
        raise ValueError(f"sample_seeds must have shape ({row_count},)")
    if int(packed_offsets[0].item()) != 0 or int(packed_offsets[-1].item()) != int(packed_scores.shape[0]):
        raise ValueError("packed offsets must describe the packed score vector exactly")

    lengths = packed_offsets[1:] - packed_offsets[:-1]
    actions = torch.full(
        (row_count,),
        int(pass_action_id),
        device=packed_scores.device,
        dtype=torch.long,
    )
    selected_logp = torch.zeros((row_count,), device=packed_scores.device, dtype=packed_scores.dtype)
    non_empty_rows = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return actions, selected_logp

    non_empty_lengths = lengths[non_empty_rows].to(dtype=torch.long)
    row_indices = packed_row_indices(packed_offsets)
    row_log_z = packed_row_log_z(packed_scores, packed_offsets)
    repeated_log_z = row_log_z.index_select(0, row_indices)
    log_probs = packed_scores - repeated_log_z
    probs = torch.exp(log_probs)
    local_cdf = packed_local_cdf_fn(probs, packed_offsets)
    thresholds = uniform_fn(
        sample_seeds.to(device=packed_scores.device, dtype=torch.long).index_select(0, non_empty_rows),
        dtype=packed_scores.dtype,
    )
    repeated_thresholds = thresholds.index_select(0, row_indices)
    previous_cdf = local_cdf - probs
    chosen = (local_cdf >= repeated_thresholds) & (previous_cdf < repeated_thresholds)
    packed_positions = torch.arange(packed_scores.shape[0], device=packed_scores.device, dtype=packed_scores.dtype)
    sentinel = torch.full_like(packed_positions, float(packed_scores.shape[0]))
    chosen_positions = torch.segment_reduce(
        torch.where(chosen, packed_positions, sentinel),
        reduce="amin",
        lengths=non_empty_lengths,
    ).to(dtype=torch.long)
    missing_rows = torch.nonzero(chosen_positions == packed_scores.shape[0], as_tuple=False).squeeze(1)
    if missing_rows.numel() > 0:
        fallback_positions = (
            packed_offsets[1:]
            .to(device=packed_scores.device, dtype=torch.long)
            .index_select(0, non_empty_rows.index_select(0, missing_rows))
            - 1
        )
        chosen_positions = chosen_positions.clone()
        chosen_positions[missing_rows] = fallback_positions
    chosen_actions = packed_ids.index_select(0, chosen_positions)
    chosen_logp = log_probs.index_select(0, chosen_positions)
    actions[non_empty_rows] = chosen_actions
    selected_logp[non_empty_rows] = chosen_logp
    return actions, selected_logp
