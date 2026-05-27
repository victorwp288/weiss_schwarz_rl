"""Deterministic model-side action sampling helpers."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from weiss_rl.models.tensor_ops import (
    packed_local_cdf,
    packed_row_indices,
    packed_row_log_z,
    uniform_from_seeds,
)


def sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
    temperature: float = 1.0,
    uniform_from_seeds_fn: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
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
    sample_log_probs = _temperature_scaled_log_probs(log_probs, mask=mask, temperature=temperature)
    probs = torch.where(mask, torch.exp(sample_log_probs), torch.zeros_like(sample_log_probs))
    cdf = torch.cumsum(probs, dim=1)
    seed_rows = sample_seeds.index_select(0, non_empty_rows).to(device=log_probs.device, dtype=torch.long)
    thresholds = (
        uniform_from_seeds(seed_rows, dtype=log_probs.dtype)
        if uniform_from_seeds_fn is None
        else uniform_from_seeds_fn(seed_rows)
    ).unsqueeze(1)
    cdf_rows = cdf.index_select(0, non_empty_rows)
    chosen = cdf_rows >= thresholds
    chosen_indices = chosen.to(dtype=torch.int64).argmax(dim=1)
    fallback_indices = mask.index_select(0, non_empty_rows).to(dtype=torch.int64).argmax(dim=1)
    chosen_indices = torch.where(chosen.any(dim=1), chosen_indices, fallback_indices)
    actions[non_empty_rows] = chosen_indices
    selected_logp[non_empty_rows] = (
        sample_log_probs.index_select(0, non_empty_rows)
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
    temperature: float = 1.0,
    packed_row_indices_fn: Callable[[Tensor], Tensor] = packed_row_indices,
    packed_row_log_z_fn: Callable[[Tensor, Tensor], Tensor] = packed_row_log_z,
    packed_local_cdf_fn: Callable[[Tensor, Tensor], Tensor] = packed_local_cdf,
    uniform_from_seeds_fn: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
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
    temperature_value = _coerce_sampling_temperature(temperature)
    sample_scores = packed_scores if temperature_value == 1.0 else packed_scores / temperature_value
    row_indices = packed_row_indices_fn(packed_offsets)
    row_log_z = packed_row_log_z_fn(sample_scores, packed_offsets)
    repeated_log_z = row_log_z.index_select(0, row_indices)
    log_probs = sample_scores - repeated_log_z
    probs = torch.exp(log_probs)
    local_cdf = packed_local_cdf_fn(probs, packed_offsets)
    seed_rows = sample_seeds.to(device=packed_scores.device, dtype=torch.long).index_select(0, non_empty_rows)
    thresholds = (
        uniform_from_seeds(seed_rows, dtype=packed_scores.dtype)
        if uniform_from_seeds_fn is None
        else uniform_from_seeds_fn(seed_rows)
    )
    repeated_thresholds = thresholds.index_select(0, row_indices)
    chosen = local_cdf >= repeated_thresholds
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


def _temperature_scaled_log_probs(log_probs: Tensor, *, mask: Tensor, temperature: float) -> Tensor:
    temperature_value = _coerce_sampling_temperature(temperature)
    if temperature_value == 1.0:
        return log_probs
    scaled = log_probs / temperature_value
    neg_inf = torch.full_like(scaled, -torch.inf)
    masked_scaled = torch.where(mask, scaled, neg_inf)
    result = neg_inf.clone()
    row_has_candidates = mask.any(dim=1)
    if bool(row_has_candidates.any().item()):
        rows = torch.nonzero(row_has_candidates, as_tuple=False).squeeze(1)
        result[rows] = torch.log_softmax(masked_scaled.index_select(0, rows), dim=1)
    return result


def _coerce_sampling_temperature(temperature: float) -> float:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    return value
