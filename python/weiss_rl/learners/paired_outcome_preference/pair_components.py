"""Shared pair-component containers and weighting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreferencePairComponents:
    margins: list[Tensor]
    pair_losses: list[Tensor]
    pair_weights: list[Tensor]
    pair_group_ids: list[int]
    current_pref_values: list[Tensor]
    current_rej_values: list[Tensor]
    incomplete_pair_count: int


def weighted_pair_loss(pair_losses: Tensor, pair_weights: Tensor) -> Tensor:
    if pair_losses.shape != pair_weights.shape:
        raise ValueError("pair_weights must have the same shape as pair_losses")
    weight_sum = pair_weights.sum()
    if bool((weight_sum <= 0.0).detach().cpu().item()):
        return pair_losses.mean()
    return (pair_losses * pair_weights).sum() / weight_sum


def balanced_pair_loss(pair_losses: Tensor, pair_group_ids: list[int], pair_weights: Tensor) -> Tensor:
    if len(pair_group_ids) != int(pair_losses.numel()):
        raise ValueError("pair_group_ids must have one item per preference pair")
    if pair_weights.shape != pair_losses.shape:
        raise ValueError("pair_weights must have the same shape as pair_losses")
    groups = sorted(set(pair_group_ids))
    group_losses = []
    group_ids_tensor = torch.as_tensor(pair_group_ids, device=pair_losses.device, dtype=torch.long)
    for group_id in groups:
        group_mask = group_ids_tensor == int(group_id)
        if bool(group_mask.any().item()):
            group_losses.append(weighted_pair_loss(pair_losses[group_mask], pair_weights[group_mask]))
    if not group_losses:
        return pair_losses.mean()
    return torch.stack(group_losses).mean()


__all__ = [
    "PreferencePairComponents",
    "balanced_pair_loss",
    "weighted_pair_loss",
]
