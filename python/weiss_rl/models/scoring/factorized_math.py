"""Tensor helpers for factorized structured-policy scoring."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.models.backbone.tensor_ops import (
    derived_sample_seeds,
    factorized_local_row_indices,
    masked_entropy_from_log_probs,
    masked_log_softmax,
    scatter_factorized_row_values,
)

_masked_log_softmax = masked_log_softmax
_masked_entropy_from_log_probs = masked_entropy_from_log_probs
_derived_sample_seeds = derived_sample_seeds
_factorized_local_row_indices = factorized_local_row_indices
_scatter_factorized_row_values = scatter_factorized_row_values


def _segment_max(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    if keys.numel() == 0:
        return out
    out.scatter_reduce_(0, keys.to(dtype=torch.long), values, reduce="amax", include_self=True)
    return out


def _segment_logsumexp(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    max_per = _segment_max(values, keys, int(num_segments))
    if keys.numel() == 0:
        return max_per
    long_keys = keys.to(dtype=torch.long)
    gathered_max = max_per.index_select(0, long_keys)
    shifted = torch.exp(values - gathered_max)
    sumexp = torch.zeros((int(num_segments),), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, long_keys, shifted)
    valid = torch.isfinite(max_per) & (sumexp > 0)
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    out[valid] = torch.log(sumexp[valid]) + max_per[valid]
    return out


def _sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # Resolve lazily through weiss_rl.model so the private sampling wrapper remains monkeypatchable.
    from weiss_rl import model as model_module

    return model_module._sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=sample_seeds,
        default_index=default_index,
        temperature=temperature,
    )


__all__ = [
    "_derived_sample_seeds",
    "_factorized_local_row_indices",
    "_masked_entropy_from_log_probs",
    "_masked_log_softmax",
    "_sample_masked_log_probs",
    "_scatter_factorized_row_values",
    "_segment_logsumexp",
    "_segment_max",
]
