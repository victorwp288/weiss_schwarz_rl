"""Retention terms for paired outcome preference repair losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def preference_retention_loss_and_metrics(
    *,
    current: Tensor,
    reference: Tensor,
    roles: Tensor,
    valid: Tensor,
    role: str,
    scope_mask: Tensor | None,
    margin: float,
    reference_best_non_target: Tensor | None,
    reference_top_only: bool,
    dtype: torch.dtype,
    metric_prefix: str,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    if role == "preferred":
        retention_mask = valid & (roles == 1)
    elif role == "rejected":
        retention_mask = valid & (roles == 0)
    else:
        retention_mask = valid
    if scope_mask is not None:
        retention_mask = retention_mask & scope_mask
    if reference_top_only:
        if reference_best_non_target is None:
            raise ValueError("reference_best_non_target_logp is required when retention_reference_top_only is enabled")
        retention_mask = (
            retention_mask & torch.isfinite(reference_best_non_target) & (reference >= reference_best_non_target)
        )
    zero = finite_graph_zero(current)
    row_count = int(retention_mask.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, empty_retention_metrics(metric_prefix=metric_prefix), {}
    logp_delta = current[retention_mask] - reference[retention_mask]
    violations = F.relu(reference[retention_mask] + float(margin) - current[retention_mask])
    retention_loss = violations.mean().to(dtype=dtype)
    violation_mask = violations > 0.0
    metrics = {
        f"{metric_prefix}_retention_loss": float(retention_loss.detach().item()),
        f"{metric_prefix}_retention_row_count": float(row_count),
        f"{metric_prefix}_retention_violation_fraction": float(violation_mask.to(dtype=dtype).mean().detach().item()),
        f"{metric_prefix}_retention_logp_delta_mean": float(logp_delta.mean().detach().item()),
        f"{metric_prefix}_retention_logp_delta_min": float(logp_delta.min().detach().item()),
        f"{metric_prefix}_retention_reference_top_only": 1.0 if reference_top_only else 0.0,
        f"{metric_prefix}_retention_scoped": 1.0 if scope_mask is not None else 0.0,
    }
    tensors = {
        f"{metric_prefix}_retention_logp_delta": logp_delta.detach(),
        f"{metric_prefix}_retention_violations": violations.detach(),
    }
    return retention_loss, metrics, tensors


def preference_top_action_retention_loss_and_metrics(
    *,
    current: Tensor,
    best_non_target: Tensor | None,
    roles: Tensor,
    valid: Tensor,
    role: str,
    scope_mask: Tensor | None,
    margin: float,
    reference: Tensor,
    reference_best_non_target: Tensor | None,
    reference_top_only: bool,
    dtype: torch.dtype,
    metric_prefix: str,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    zero = finite_graph_zero(current)
    if best_non_target is None:
        return zero, empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
    if role == "preferred":
        retention_mask = valid & (roles == 1)
    elif role == "rejected":
        retention_mask = valid & (roles == 0)
    else:
        retention_mask = valid
    if scope_mask is not None:
        retention_mask = retention_mask & scope_mask
    retention_mask = retention_mask & torch.isfinite(best_non_target)
    if reference_top_only:
        if reference_best_non_target is None:
            raise ValueError(
                "reference_best_non_target_logp is required when top_action_retention_reference_top_only is enabled"
            )
        retention_mask = (
            retention_mask & torch.isfinite(reference_best_non_target) & (reference >= reference_best_non_target)
        )
    row_count = int(retention_mask.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
    gap = current[retention_mask] - best_non_target[retention_mask]
    violations = F.relu(best_non_target[retention_mask] + float(margin) - current[retention_mask])
    retention_loss = violations.mean().to(dtype=dtype)
    violation_mask = violations > 0.0
    metrics = {
        f"{metric_prefix}_top_action_retention_loss": float(retention_loss.detach().item()),
        f"{metric_prefix}_top_action_retention_row_count": float(row_count),
        f"{metric_prefix}_top_action_retention_violation_fraction": float(
            violation_mask.to(dtype=dtype).mean().detach().item()
        ),
        f"{metric_prefix}_top_action_retention_gap_mean": float(gap.mean().detach().item()),
        f"{metric_prefix}_top_action_retention_gap_min": float(gap.min().detach().item()),
        f"{metric_prefix}_top_action_retention_reference_top_only": 1.0 if reference_top_only else 0.0,
        f"{metric_prefix}_top_action_retention_scoped": 1.0 if scope_mask is not None else 0.0,
    }
    tensors = {
        f"{metric_prefix}_top_action_retention_gap": gap.detach(),
        f"{metric_prefix}_top_action_retention_violations": violations.detach(),
    }
    return retention_loss, metrics, tensors


def empty_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_retention_loss": 0.0,
        f"{metric_prefix}_retention_row_count": 0.0,
        f"{metric_prefix}_retention_violation_fraction": 0.0,
        f"{metric_prefix}_retention_logp_delta_mean": 0.0,
        f"{metric_prefix}_retention_logp_delta_min": 0.0,
    }


def empty_top_action_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_top_action_retention_loss": 0.0,
        f"{metric_prefix}_top_action_retention_row_count": 0.0,
        f"{metric_prefix}_top_action_retention_violation_fraction": 0.0,
        f"{metric_prefix}_top_action_retention_gap_mean": 0.0,
        f"{metric_prefix}_top_action_retention_gap_min": 0.0,
    }


def finite_graph_zero(values: Tensor) -> Tensor:
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return finite_values.sum() * 0.0


__all__ = [
    "empty_retention_metrics",
    "empty_top_action_retention_metrics",
    "finite_graph_zero",
    "preference_retention_loss_and_metrics",
    "preference_top_action_retention_loss_and_metrics",
]
