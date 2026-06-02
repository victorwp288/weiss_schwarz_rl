"""Top-action retention helpers for paired-swing replay repair."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True)
class PairedSwingTopActionRetentionRows:
    gaps: Tensor
    weights: Tensor
    agreements: Tensor


def paired_swing_top_action_retention_rows(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor,
    legal_offsets: Tensor,
    supported: Tensor,
    supported_weight: Tensor,
) -> PairedSwingTopActionRetentionRows | None:
    row_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    if int(row_indices.numel()) <= 0:
        return None

    offsets = legal_offsets.to(device=packed_logits.device, dtype=torch.long)
    row_weights = supported_weight.to(device=packed_logits.device, dtype=torch.float32)
    gaps: list[Tensor] = []
    weights: list[Tensor] = []
    agreements: list[Tensor] = []
    for row_index_tensor in row_indices:
        row_index = int(row_index_tensor.detach().cpu().item())
        start = int(offsets[row_index].detach().cpu().item())
        stop = int(offsets[row_index + 1].detach().cpu().item())
        if stop <= start + 1:
            continue
        current_row = torch.log_softmax(packed_logits[start:stop], dim=0)
        reference_row = torch.log_softmax(reference_packed_logits[start:stop], dim=0)
        reference_top_offset = int(torch.argmax(reference_row).detach().cpu().item())
        current_top_offset = int(torch.argmax(current_row).detach().cpu().item())
        current_reference_top = current_row[reference_top_offset]
        current_best_other = current_row.masked_fill(
            torch.arange(stop - start, device=packed_logits.device) == reference_top_offset,
            float("-inf"),
        ).max()
        if not bool(torch.isfinite(current_reference_top).item()) or not bool(
            torch.isfinite(current_best_other).item()
        ):
            continue
        gaps.append(current_reference_top - current_best_other)
        weights.append(row_weights[row_index])
        agreements.append(
            torch.as_tensor(float(current_top_offset == reference_top_offset), device=packed_logits.device)
        )
    if not gaps:
        return None
    return PairedSwingTopActionRetentionRows(
        gaps=torch.stack(gaps).to(dtype=packed_logits.dtype),
        weights=torch.stack(weights).to(device=packed_logits.device, dtype=torch.float32),
        agreements=torch.stack(agreements).to(device=packed_logits.device, dtype=torch.float32),
    )


def paired_swing_top_action_retention_loss_and_metrics(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor | None,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    supported: Tensor,
    supported_weight: Tensor,
    retention_margin: float,
    metric_prefix: str,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    del legal_ids
    zero = packed_logits.sum() * 0.0
    if reference_packed_logits is None:
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}

    rows = paired_swing_top_action_retention_rows(
        packed_logits=packed_logits,
        reference_packed_logits=reference_packed_logits,
        legal_offsets=legal_offsets,
        supported=supported,
        supported_weight=supported_weight,
    )
    if rows is None:
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}

    violations = torch.relu(rows.gaps.new_tensor(float(retention_margin)) - rows.gaps)
    loss = weighted_mean(violations, rows.weights).to(dtype=packed_logits.dtype)
    violation_fraction = weighted_mean((violations > 0.0).to(dtype=rows.weights.dtype), rows.weights)
    agreement_fraction = weighted_mean(rows.agreements, rows.weights)
    metrics = {
        f"{metric_prefix}_top_action_retention_loss": float(loss.detach().item()),
        f"{metric_prefix}_top_action_retention_rows": float(rows.gaps.numel()),
        f"{metric_prefix}_top_action_retention_violation_fraction": float(violation_fraction.detach().item()),
        f"{metric_prefix}_top_action_retention_gap_mean": float(weighted_mean(rows.gaps, rows.weights).detach().item()),
        f"{metric_prefix}_top_action_retention_gap_min": float(rows.gaps.detach().min().item()),
        f"{metric_prefix}_top_action_retention_agreement_fraction": float(agreement_fraction.detach().item()),
    }
    return loss, metrics, {f"{metric_prefix}_top_action_retention_gap": rows.gaps.detach()}


def _empty_top_action_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_top_action_retention_loss": 0.0,
        f"{metric_prefix}_top_action_retention_rows": 0.0,
        f"{metric_prefix}_top_action_retention_violation_fraction": 0.0,
        f"{metric_prefix}_top_action_retention_gap_mean": 0.0,
        f"{metric_prefix}_top_action_retention_gap_min": 0.0,
        f"{metric_prefix}_top_action_retention_agreement_fraction": 0.0,
    }


__all__ = [
    "PairedSwingTopActionRetentionRows",
    "paired_swing_top_action_retention_loss_and_metrics",
    "paired_swing_top_action_retention_rows",
]
