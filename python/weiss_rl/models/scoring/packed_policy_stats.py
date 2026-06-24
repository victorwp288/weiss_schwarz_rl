"""Packed legal-action policy statistics for factorized model evaluation."""

from __future__ import annotations

import torch
from torch import Tensor


def packed_log_prob_policy_stats(
    packed_log_probs: Tensor,
    *,
    legal_action_ids: Tensor,
    legal_action_offsets: Tensor,
    actions: Tensor | None = None,
) -> tuple[Tensor | None, Tensor, Tensor]:
    if packed_log_probs.ndim != 1:
        raise ValueError(f"packed_log_probs must be 1D, got shape {tuple(packed_log_probs.shape)}")
    ids = legal_action_ids.to(device=packed_log_probs.device, dtype=torch.long)
    offsets = legal_action_offsets.to(device=packed_log_probs.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    lengths = offsets[1:] - offsets[:-1]
    if int(ids.numel()) != int(packed_log_probs.numel()) or int(lengths.sum().item()) != int(
        packed_log_probs.numel()
    ):
        raise ValueError("packed legal ids/offsets must align with packed log-probs")

    row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_log_probs.device), lengths)
    entropy = torch.zeros((row_count,), device=packed_log_probs.device, dtype=packed_log_probs.dtype)
    if int(packed_log_probs.numel()) > 0:
        safe_log_probs = torch.where(
            torch.isfinite(packed_log_probs),
            packed_log_probs,
            torch.zeros_like(packed_log_probs),
        )
        entropy.scatter_add_(0, row_indices, -(torch.exp(packed_log_probs) * safe_log_probs))

    top_action_ids = torch.full((row_count,), -1, device=packed_log_probs.device, dtype=torch.long)
    if int(packed_log_probs.numel()) > 0:
        row_max = torch.full((row_count,), -torch.inf, device=packed_log_probs.device, dtype=packed_log_probs.dtype)
        row_max.scatter_reduce_(0, row_indices, packed_log_probs, reduce="amax", include_self=True)
        max_for_candidate = row_max.index_select(0, row_indices)
        sentinel = torch.iinfo(torch.long).max
        top_candidates = torch.where(
            packed_log_probs == max_for_candidate,
            ids,
            torch.full_like(ids, sentinel),
        )
        top_min = torch.full((row_count,), sentinel, device=packed_log_probs.device, dtype=torch.long)
        top_min.scatter_reduce_(0, row_indices, top_candidates, reduce="amin", include_self=True)
        top_action_ids = torch.where(top_min == sentinel, top_action_ids, top_min)

    action_logp = None
    if actions is not None:
        flat_actions = actions.reshape(-1).to(device=packed_log_probs.device, dtype=torch.long)
        if int(flat_actions.numel()) != row_count:
            raise ValueError(f"actions must have length {row_count}, got {int(flat_actions.numel())}")
        action_logp = torch.full(
            (row_count,),
            -torch.inf,
            device=packed_log_probs.device,
            dtype=packed_log_probs.dtype,
        )
        if int(packed_log_probs.numel()) > 0:
            candidate_actions = flat_actions.index_select(0, row_indices)
            selected_values = torch.where(
                ids == candidate_actions,
                packed_log_probs,
                torch.full_like(packed_log_probs, -torch.inf),
            )
            action_logp.scatter_reduce_(0, row_indices, selected_values, reduce="amax", include_self=True)
    return action_logp, entropy, top_action_ids


__all__ = ["packed_log_prob_policy_stats"]
