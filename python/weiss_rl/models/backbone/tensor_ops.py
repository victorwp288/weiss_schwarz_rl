"""Pure tensor helpers used by policy/value model scoring paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def bucket_card_ids(card_ids: Tensor, *, vocab_size: int) -> Tensor:
    if vocab_size <= 1:
        return torch.zeros_like(card_ids, dtype=torch.long)
    card_ids_long = card_ids.to(dtype=torch.long)
    positive_ids = torch.where(card_ids_long > 0, card_ids_long, torch.zeros_like(card_ids_long))
    hashed = torch.remainder(positive_ids, vocab_size - 1) + 1
    return torch.where(positive_ids > 0, hashed, torch.zeros_like(hashed))


def masked_mean_pool(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=values.dtype)
    total = (values * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return total / denom


def masked_max_pool(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[2]))
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    pooled = masked.max(dim=1).values
    has_any = mask.any(dim=1, keepdim=True)
    return torch.where(has_any, pooled, torch.zeros_like(pooled))


def optional_embedding(embedding: nn.Embedding, indices: Tensor) -> Tensor:
    safe_ids = torch.where(indices >= 0, indices + 1, torch.zeros_like(indices))
    return embedding(safe_ids.to(dtype=torch.long))


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
    cdf_dtype = (
        torch.float64 if probabilities.dtype in {torch.float16, torch.bfloat16, torch.float32} else probabilities.dtype
    )
    probabilities_for_cdf = probabilities.to(dtype=cdf_dtype)
    row_count = int(offsets.shape[0] - 1)
    row_indices = packed_row_indices(offsets)
    cumulative = torch.cumsum(probabilities_for_cdf, dim=0)
    base = torch.zeros((row_count,), dtype=cdf_dtype, device=probabilities.device)
    if row_count > 1:
        starts = offsets[1:-1].to(dtype=torch.long)
        base[1:] = cumulative.index_select(0, starts - 1)
    return (cumulative - base.index_select(0, row_indices)).to(dtype=probabilities.dtype)


def uniform_from_seeds(sample_seeds: Tensor, *, dtype: torch.dtype) -> Tensor:
    seed_float = sample_seeds.to(dtype=torch.float64)
    hashed = torch.sin(seed_float * 12.9898 + 78.233) * 43758.5453123
    uniform = torch.remainder(hashed, 1.0).to(dtype=dtype)
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


def factorized_local_row_indices(available_rows: Tensor, selected_rows: Tensor) -> Tensor:
    if selected_rows.numel() == 0:
        return selected_rows.new_zeros((0,), dtype=torch.long)
    if available_rows.numel() == 0:
        raise ValueError("factorized row lookup requires at least one available row")
    positions = torch.searchsorted(available_rows, selected_rows)
    if bool((positions >= available_rows.shape[0]).any().item()):
        raise ValueError("factorized row lookup exceeded available rows")
    matched_rows = available_rows.index_select(0, positions)
    if not bool(torch.equal(matched_rows, selected_rows)):
        raise ValueError("factorized row lookup requires selected rows to be legal for the chosen family")
    return positions


def scatter_factorized_row_values(
    row_count: int,
    row_indices: Tensor,
    values: Tensor,
    *,
    fill_value: float = -torch.inf,
) -> Tensor:
    output = values.new_full((row_count, *values.shape[1:]), fill_value)
    if row_indices.numel() > 0:
        output.index_copy_(0, row_indices.to(dtype=torch.long), values)
    return output
