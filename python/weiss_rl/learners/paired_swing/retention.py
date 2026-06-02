"""Retention losses for paired-swing replay repair."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_swing.margin_retention import paired_swing_margin_retention_loss_and_metrics
from weiss_rl.learners.paired_swing.rows import positive_vs_top_other_margin_by_row
from weiss_rl.learners.paired_swing.top_retention import paired_swing_top_action_retention_loss_and_metrics
from weiss_rl.learners.tensor_ops import weighted_mean


def packed_top_action_retention_loss(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor | None,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    loss_mask: Tensor,
    retention_margin: float = 0.0,
    metric_prefix: str = "paired_swing_full_surface",
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Retain the reference model's legal top action on every masked row."""

    if reference_packed_logits is not None and reference_packed_logits.shape != packed_logits.shape:
        raise ValueError("reference_packed_logits must match packed_logits shape")
    if float(retention_margin) < 0.0:
        raise ValueError("top-action retention margin must be >= 0")
    offsets = legal_offsets.to(device=packed_logits.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    if row_count < 0:
        raise ValueError("legal_offsets must contain at least one offset")
    if int(loss_mask.numel()) != row_count:
        raise ValueError(f"loss_mask row count {int(loss_mask.numel())} does not match packed row count {row_count}")
    if row_count > 0 and int(offsets[-1].detach().cpu().item()) != int(packed_logits.numel()):
        raise ValueError("legal_offsets do not match packed_logits length")
    flat_loss_mask = loss_mask.reshape(-1).to(device=packed_logits.device, dtype=torch.float32)
    supported = flat_loss_mask > 0.0
    return paired_swing_top_action_retention_loss_and_metrics(
        packed_logits=packed_logits,
        reference_packed_logits=reference_packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        supported=supported,
        supported_weight=flat_loss_mask,
        retention_margin=float(retention_margin),
        metric_prefix=metric_prefix,
    )


def packed_target_action_retention_loss(
    *,
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    target_actions: Tensor,
    target_valid: Tensor | None,
    loss_mask: Tensor,
    retention_margin: float = 0.0,
    metric_prefix: str = "paired_swing_full_surface_target",
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Require each masked target action to remain ahead of the best other legal action."""

    if target_actions.shape != loss_mask.shape:
        raise ValueError("target_actions must match loss_mask shape")
    if target_valid is not None and target_valid.shape != loss_mask.shape:
        raise ValueError("target_valid must match loss_mask shape")
    if float(retention_margin) < 0.0:
        raise ValueError("target-action retention margin must be >= 0")
    offsets = legal_offsets.to(device=packed_logits.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    if row_count < 0:
        raise ValueError("legal_offsets must contain at least one offset")
    if int(loss_mask.numel()) != row_count:
        raise ValueError(f"loss_mask row count {int(loss_mask.numel())} does not match packed row count {row_count}")
    if row_count > 0 and int(offsets[-1].detach().cpu().item()) != int(packed_logits.numel()):
        raise ValueError("legal_offsets do not match packed_logits length")

    flat_loss_mask = loss_mask.reshape(-1).to(device=packed_logits.device, dtype=torch.float32)
    flat_target_actions = target_actions.reshape(-1).to(device=packed_logits.device, dtype=torch.long)
    flat_target_valid = (
        torch.ones_like(flat_target_actions, dtype=torch.bool)
        if target_valid is None
        else target_valid.reshape(-1).to(device=packed_logits.device, dtype=torch.bool)
    )
    active_rows = (flat_loss_mask > 0.0) & flat_target_valid & (flat_target_actions >= 0)
    margin_by_row, supported, target_logp_by_row, best_other_logp_by_row = positive_vs_top_other_margin_by_row(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        flat_positive_actions=flat_target_actions,
        active_rows=active_rows,
    )
    zero = packed_logits.sum() * 0.0
    row_count = int(supported.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, _empty_target_action_retention_metrics(metric_prefix=metric_prefix), {}
    weights = flat_loss_mask[supported]
    margins = margin_by_row[supported].to(dtype=packed_logits.dtype)
    violations = torch.relu(margins.new_tensor(float(retention_margin)) - margins)
    loss = weighted_mean(violations, weights).to(dtype=packed_logits.dtype)
    violation_fraction = weighted_mean((violations > 0.0).to(dtype=weights.dtype), weights)
    top_fraction = weighted_mean((margins >= 0.0).to(dtype=weights.dtype), weights)
    metrics = {
        f"{metric_prefix}_retention_loss": float(loss.detach().item()),
        f"{metric_prefix}_retention_rows": float(row_count),
        f"{metric_prefix}_retention_violation_fraction": float(violation_fraction.detach().item()),
        f"{metric_prefix}_retention_margin_mean": float(weighted_mean(margins, weights).detach().item()),
        f"{metric_prefix}_retention_margin_min": float(margins.detach().min().item()),
        f"{metric_prefix}_retention_target_top_fraction": float(top_fraction.detach().item()),
        f"{metric_prefix}_retention_target_logp_mean": float(
            weighted_mean(target_logp_by_row[supported], weights).detach().item()
        ),
        f"{metric_prefix}_retention_best_other_logp_mean": float(
            weighted_mean(best_other_logp_by_row[supported], weights).detach().item()
        ),
    }
    return loss, metrics, {f"{metric_prefix}_retention_margin": margins.detach()}


def _empty_target_action_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_retention_loss": 0.0,
        f"{metric_prefix}_retention_rows": 0.0,
        f"{metric_prefix}_retention_violation_fraction": 0.0,
        f"{metric_prefix}_retention_margin_mean": 0.0,
        f"{metric_prefix}_retention_margin_min": 0.0,
        f"{metric_prefix}_retention_target_top_fraction": 0.0,
        f"{metric_prefix}_retention_target_logp_mean": 0.0,
        f"{metric_prefix}_retention_best_other_logp_mean": 0.0,
    }


__all__ = [
    "packed_target_action_retention_loss",
    "packed_top_action_retention_loss",
    "paired_swing_margin_retention_loss_and_metrics",
    "paired_swing_top_action_retention_loss_and_metrics",
]
