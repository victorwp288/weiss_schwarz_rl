"""Scope-specific objective aggregation for paired-swing replay losses."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


def paired_swing_scoped_margin_loss(
    *,
    margins: Tensor,
    supported_weight: Tensor,
    supported: Tensor,
    positive_actions: Tensor,
    group_ids: Tensor | None,
    normalized_scope: str,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    if normalized_scope == "episode_mean":
        return _episode_mean_margin_loss(
            margins=margins,
            supported_weight=supported_weight,
            supported=supported,
            episode_count=int(positive_actions.shape[1]),
            margin=float(margin),
        )
    if normalized_scope == "label_mean":
        if group_ids is None:
            raise ValueError("paired_swing loss_scope label_mean requires group_ids")
        return _group_mean_margin_loss(
            margins=margins,
            group_ids=group_ids,
            margin=float(margin),
        )
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss = weighted_mean(violations, supported_weight).to(dtype=margins.dtype)
    margin_mean = weighted_mean(margins, supported_weight)
    satisfied_fraction = (
        (margins >= float(margin)).to(dtype=supported_weight.dtype) * supported_weight
    ).sum() / torch.clamp(supported_weight.sum(), min=1.0e-8)
    return loss, margin_mean, satisfied_fraction, {}


def _group_mean_margin_loss(
    *,
    margins: Tensor,
    group_ids: Tensor,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    if group_ids.shape != margins.shape:
        raise ValueError("paired-swing group_ids must match supported margins")
    unique_group_ids = torch.unique(group_ids, sorted=True)
    group_margins: list[Tensor] = []
    for group_id in unique_group_ids:
        group_mask = group_ids == group_id
        group_margins.append(margins[group_mask].mean())
    if not group_margins:
        zero = margins.sum() * 0.0
        return zero, zero, zero, {"paired_swing_label_count": 0.0, "paired_swing_label_rows_mean": 0.0}
    stacked_margins = torch.stack(group_margins).to(dtype=margins.dtype)
    violations = torch.relu(stacked_margins.new_tensor(float(margin)) - stacked_margins)
    loss = violations.mean().to(dtype=margins.dtype)
    margin_mean = stacked_margins.mean()
    satisfied_fraction = (stacked_margins >= float(margin)).to(dtype=margins.dtype).mean()
    return (
        loss,
        margin_mean,
        satisfied_fraction,
        {
            "paired_swing_label_count": float(unique_group_ids.numel()),
            "paired_swing_label_rows_mean": float(margins.numel() / max(int(unique_group_ids.numel()), 1)),
        },
    )


def _episode_mean_margin_loss(
    *,
    margins: Tensor,
    supported_weight: Tensor,
    supported: Tensor,
    episode_count: int,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    supported_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    episode_ids = torch.remainder(supported_indices, int(episode_count))
    unique_episode_ids = torch.unique(episode_ids, sorted=True)
    episode_margins: list[Tensor] = []
    episode_weights: list[Tensor] = []
    for episode_id in unique_episode_ids:
        episode_mask = episode_ids == episode_id
        weights = supported_weight[episode_mask]
        episode_margins.append(weighted_mean(margins[episode_mask], weights))
        episode_weights.append(weights.sum())
    stacked_margins = torch.stack(episode_margins).to(dtype=margins.dtype)
    stacked_weights = torch.stack(episode_weights).to(dtype=supported_weight.dtype)
    violations = torch.relu(stacked_margins.new_tensor(float(margin)) - stacked_margins)
    loss = weighted_mean(violations, stacked_weights).to(dtype=margins.dtype)
    margin_mean = weighted_mean(stacked_margins, stacked_weights)
    satisfied_fraction = (
        (stacked_margins >= float(margin)).to(dtype=stacked_weights.dtype) * stacked_weights
    ).sum() / torch.clamp(stacked_weights.sum(), min=1.0e-8)
    return (
        loss,
        margin_mean,
        satisfied_fraction,
        {
            "paired_swing_episode_count": float(unique_episode_ids.numel()),
            "paired_swing_episode_rows_mean": float(
                (supported_weight.sum() / max(int(unique_episode_ids.numel()), 1)).detach().item()
            ),
        },
    )


__all__ = ["paired_swing_scoped_margin_loss"]
