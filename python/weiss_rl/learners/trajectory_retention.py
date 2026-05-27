"""Auxiliary losses for preserving selected trajectory behavior."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


def trajectory_retention_action_loss(
    *,
    action_logp: Tensor,
    actions: Tensor,
    retention_valid: Tensor | None,
    coef: float,
    top_action_ids: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Imitate retained behavior actions without adding rows to policy-gradient loss."""

    zero = action_logp.sum() * 0.0
    coef_f = float(coef)
    if coef_f <= 0.0:
        return zero, {}

    metrics: dict[str, float] = {"trajectory_retention_coef_active": coef_f}
    if retention_valid is None:
        metrics.update(
            {
                "trajectory_retention_valid_fraction": 0.0,
                "trajectory_retention_supported_fraction": 0.0,
                "trajectory_retention_loss": 0.0,
                "trajectory_retention_weighted_loss": 0.0,
            }
        )
        return zero, metrics

    if tuple(retention_valid.shape) != tuple(action_logp.shape):
        raise ValueError(
            "trajectory_retention_valid must match action_logp shape "
            f"{tuple(action_logp.shape)}, got {tuple(retention_valid.shape)}"
        )
    if tuple(actions.shape) != tuple(action_logp.shape):
        raise ValueError(f"actions must match action_logp shape {tuple(action_logp.shape)}, got {tuple(actions.shape)}")

    valid = retention_valid.to(device=action_logp.device, dtype=torch.bool)
    metrics["trajectory_retention_valid_fraction"] = float(valid.to(dtype=torch.float32).mean().detach().item())
    valid_count = float(valid.to(dtype=torch.float32).sum().detach().item())
    supported = valid & torch.isfinite(action_logp)
    supported_count = float(supported.to(dtype=torch.float32).sum().detach().item())
    metrics["trajectory_retention_supported_fraction"] = (
        0.0 if valid_count <= 0.0 else float(supported_count / max(valid_count, 1.0))
    )
    metrics["trajectory_retention_rows"] = supported_count
    if supported_count <= 0.0:
        metrics.update(
            {
                "trajectory_retention_loss": 0.0,
                "trajectory_retention_weighted_loss": 0.0,
            }
        )
        return zero, metrics

    weights = supported.to(dtype=torch.float32)
    supported_logp = torch.where(supported, action_logp.to(dtype=torch.float32), torch.zeros_like(action_logp))
    nll = -supported_logp
    raw_loss = weighted_mean(nll, weights).to(dtype=action_logp.dtype)
    weighted_loss = raw_loss * coef_f
    metrics["trajectory_retention_loss"] = float(raw_loss.detach().item())
    metrics["trajectory_retention_weighted_loss"] = float(weighted_loss.detach().item())
    metrics["trajectory_retention_logp_mean"] = float(weighted_mean(supported_logp, weights).detach().item())
    if top_action_ids is not None:
        if tuple(top_action_ids.shape) != tuple(action_logp.shape):
            raise ValueError(
                "top_action_ids must match action_logp shape "
                f"{tuple(action_logp.shape)}, got {tuple(top_action_ids.shape)}"
            )
        top_ids = top_action_ids.to(device=action_logp.device, dtype=torch.long)
        action_ids = actions.to(device=action_logp.device, dtype=torch.long)
        agreement = weighted_mean((top_ids == action_ids).to(dtype=torch.float32), weights)
        metrics["trajectory_retention_top_action_accuracy"] = float(agreement.detach().item())
    return weighted_loss, metrics


__all__ = ["trajectory_retention_action_loss"]
