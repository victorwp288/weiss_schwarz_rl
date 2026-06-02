"""Margin-retention loss for paired-swing replay repair."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_swing.comparison import paired_swing_margin_comparison_rows
from weiss_rl.learners.tensor_ops import weighted_mean


def paired_swing_margin_retention_loss_and_metrics(
    *,
    current_margin_by_row: Tensor,
    reference_packed_logits: Tensor | None,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    positive_actions: Tensor,
    negative_actions: Tensor,
    flat_positive_actions: Tensor,
    flat_negative_actions: Tensor,
    active_rows: Tensor,
    supported: Tensor,
    supported_weight: Tensor,
    pass_action_id: int | None,
    compare_to: str,
    retention_margin: float,
    metric_prefix: str,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    zero = current_margin_by_row.sum() * 0.0
    if reference_packed_logits is None:
        return zero, _empty_margin_retention_metrics(metric_prefix=metric_prefix), {}

    reference_comparison = paired_swing_margin_comparison_rows(
        packed_logits=reference_packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        flat_positive_actions=flat_positive_actions,
        flat_negative_actions=flat_negative_actions,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        active_rows=active_rows,
        pass_action_id=pass_action_id,
        compare_to=compare_to,
    )
    reference_margin_by_row = reference_comparison.margin_by_row.to(dtype=current_margin_by_row.dtype)
    reference_supported = reference_comparison.supported
    retention_supported = supported & reference_supported & torch.isfinite(current_margin_by_row)
    row_count = int(retention_supported.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, _empty_margin_retention_metrics(metric_prefix=metric_prefix), {}

    margin_delta = current_margin_by_row[retention_supported] - reference_margin_by_row[retention_supported]
    violations = torch.relu(margin_delta.new_tensor(float(retention_margin)) - margin_delta)
    weights = supported_weight[retention_supported]
    loss = weighted_mean(violations, weights).to(dtype=current_margin_by_row.dtype)
    weighted_violation_fraction = weighted_mean((violations > 0.0).to(dtype=weights.dtype), weights)
    metrics = {
        f"{metric_prefix}_margin_retention_loss": float(loss.detach().item()),
        f"{metric_prefix}_margin_retention_rows": float(row_count),
        f"{metric_prefix}_margin_retention_violation_fraction": float(weighted_violation_fraction.detach().item()),
        f"{metric_prefix}_margin_delta_mean": float(weighted_mean(margin_delta, weights).detach().item()),
        f"{metric_prefix}_margin_delta_min": float(margin_delta.detach().min().item()),
    }
    return loss, metrics, {"paired_swing_margin_delta": margin_delta.detach()}


def _empty_margin_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_margin_retention_loss": 0.0,
        f"{metric_prefix}_margin_retention_rows": 0.0,
        f"{metric_prefix}_margin_retention_violation_fraction": 0.0,
        f"{metric_prefix}_margin_delta_mean": 0.0,
        f"{metric_prefix}_margin_delta_min": 0.0,
    }


__all__ = ["paired_swing_margin_retention_loss_and_metrics"]
