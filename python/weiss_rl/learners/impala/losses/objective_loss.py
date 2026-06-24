"""IMPALA objective loss assembly."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.trajectory_retention import trajectory_retention_action_loss


@dataclass(frozen=True, slots=True)
class ImpalaObjectiveLosses:
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_mean: Tensor
    value_loss_mask: Tensor
    trajectory_retention_loss: Tensor
    trajectory_retention_metrics: dict[str, float]


def compute_impala_objective_losses(
    *,
    policy_action_logp: Tensor,
    retention_action_logp: Tensor,
    actions: Tensor,
    advantages: Tensor,
    values: Tensor,
    targets: Tensor,
    entropy: Tensor,
    loss_mask: Tensor,
    value_loss_mask: Tensor | None,
    value_loss_coef: float,
    entropy_coef: float,
    trajectory_retention_valid: Tensor | None,
    trajectory_retention_coef: float,
    top_action_ids: Tensor | None = None,
) -> ImpalaObjectiveLosses:
    trajectory_retention_loss, trajectory_retention_metrics = trajectory_retention_action_loss(
        action_logp=retention_action_logp,
        actions=actions,
        retention_valid=trajectory_retention_valid,
        coef=float(trajectory_retention_coef),
        top_action_ids=top_action_ids,
    )
    resolved_value_loss_mask = value_loss_mask if value_loss_mask is not None else torch.ones_like(loss_mask)
    policy_loss_denominator = torch.clamp(loss_mask.sum(), min=1.0)
    value_loss_denominator = torch.clamp(resolved_value_loss_mask.sum(), min=1.0)

    policy_loss = -((policy_action_logp * advantages) * loss_mask).sum() / policy_loss_denominator
    value_loss = (((values - targets) ** 2) * resolved_value_loss_mask).sum() / value_loss_denominator
    entropy_mean = (entropy * loss_mask).sum() / policy_loss_denominator
    total_loss = policy_loss + (float(value_loss_coef) * value_loss) - (float(entropy_coef) * entropy_mean)
    if float(trajectory_retention_coef) != 0.0:
        total_loss = total_loss + trajectory_retention_loss

    return ImpalaObjectiveLosses(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        value_loss_mask=resolved_value_loss_mask,
        trajectory_retention_loss=trajectory_retention_loss,
        trajectory_retention_metrics=trajectory_retention_metrics,
    )


__all__ = ["ImpalaObjectiveLosses", "compute_impala_objective_losses"]
