"""Public heuristic score composition and optional logit bias."""

from __future__ import annotations

import torch
from torch import Tensor


def combine_public_heuristic_scores(
    score0: Tensor,
    score1: Tensor,
    score2: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Combine the three packed public-heuristic score components."""

    return score0.to(dtype=dtype) * 32.0 + score1.to(dtype=dtype) + (score2.to(dtype=dtype) / 4.0)


def apply_public_heuristic_bias(
    scores: Tensor,
    raw_scores: Tensor,
    *,
    scale: float,
    family_ids: Tensor | None,
    bias_family_ids: Tensor,
) -> Tensor:
    """Apply optional public-heuristic logit bias with an optional family allow-list."""

    if scale <= 0.0 or raw_scores.numel() == 0:
        return scores
    bias = raw_scores.to(dtype=scores.dtype) * (float(scale) / 100.0)
    if family_ids is None or bias_family_ids.numel() == 0:
        return scores + bias
    allowed = torch.isin(
        family_ids.to(device=bias_family_ids.device, dtype=torch.long),
        bias_family_ids,
    ).to(device=scores.device, dtype=scores.dtype)
    return scores + (bias * allowed)


__all__ = [
    "apply_public_heuristic_bias",
    "combine_public_heuristic_scores",
]
