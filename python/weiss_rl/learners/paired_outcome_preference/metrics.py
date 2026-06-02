"""Metric and tensor assembly for paired-outcome preference losses."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.pair_components import PreferencePairComponents
from weiss_rl.learners.paired_outcome_preference.retention import (
    empty_retention_metrics,
    empty_top_action_retention_metrics,
)


def paired_outcome_preference_metrics_and_tensors(
    *,
    loss: Tensor,
    pair_components: PreferencePairComponents,
    margin_tensor: Tensor,
    pair_weight_tensor: Tensor,
    candidate_pair_count: int,
    valid_row_count: int,
    metric_prefix: str,
    beta: float,
    aggregation: str,
    group_balance: bool,
    pair_weighted: bool,
    retention_coef: float,
    retention_margin: float,
    retention_role: str,
    retention_reference_top_only: bool,
    retention_scoped: bool,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    top_action_retention_role: str,
    top_action_retention_reference_top_only: bool,
    top_action_retention_scoped: bool,
    retention_metrics: dict[str, float],
    top_retention_metrics: dict[str, float],
    retention_tensors: dict[str, Tensor],
    top_retention_tensors: dict[str, Tensor],
) -> tuple[dict[str, float], dict[str, Tensor]]:
    satisfied = (margin_tensor > 0.0).to(dtype=margin_tensor.dtype)
    current_pref_tensor = torch.stack(pair_components.current_pref_values).to(dtype=margin_tensor.dtype)
    current_rej_tensor = torch.stack(pair_components.current_rej_values).to(dtype=margin_tensor.dtype)
    metrics = {
        f"{metric_prefix}_loss": float(loss.detach().item()),
        f"{metric_prefix}_pair_count": float(margin_tensor.numel()),
        f"{metric_prefix}_edge_count": float(margin_tensor.numel()) if aggregation == "edge_mean" else 0.0,
        f"{metric_prefix}_candidate_pair_count": float(candidate_pair_count),
        f"{metric_prefix}_incomplete_pair_count": float(pair_components.incomplete_pair_count),
        f"{metric_prefix}_valid_rows": float(valid_row_count),
        f"{metric_prefix}_margin_mean": float(margin_tensor.mean().detach().item()),
        f"{metric_prefix}_margin_min": float(margin_tensor.min().detach().item()),
        f"{metric_prefix}_satisfied_fraction": float(satisfied.mean().detach().item()),
        f"{metric_prefix}_current_preferred_logp_mean": float(current_pref_tensor.mean().detach().item()),
        f"{metric_prefix}_current_rejected_logp_mean": float(current_rej_tensor.mean().detach().item()),
        f"{metric_prefix}_beta": float(beta),
        f"{metric_prefix}_aggregation_sum": 1.0 if aggregation == "sum" else 0.0,
        f"{metric_prefix}_aggregation_edge_mean": 1.0 if aggregation == "edge_mean" else 0.0,
        f"{metric_prefix}_group_balance": 1.0 if group_balance else 0.0,
        f"{metric_prefix}_group_count": float(len(set(pair_components.pair_group_ids))) if group_balance else 0.0,
        f"{metric_prefix}_pair_weighted": 1.0 if pair_weighted else 0.0,
        f"{metric_prefix}_pair_weight_mean": float(pair_weight_tensor.mean().detach().item()),
        f"{metric_prefix}_pair_weight_min": float(pair_weight_tensor.min().detach().item()),
        f"{metric_prefix}_pair_weight_max": float(pair_weight_tensor.max().detach().item()),
        f"{metric_prefix}_pair_weight_nondefault_count": float(
            torch.count_nonzero(torch.abs(pair_weight_tensor - 1.0) > 1e-12).detach().item()
        ),
        f"{metric_prefix}_retention_coef": float(retention_coef),
        f"{metric_prefix}_retention_margin": float(retention_margin),
        f"{metric_prefix}_retention_role_all": 1.0 if retention_role == "all" else 0.0,
        f"{metric_prefix}_retention_role_preferred": 1.0 if retention_role == "preferred" else 0.0,
        f"{metric_prefix}_retention_role_rejected": 1.0 if retention_role == "rejected" else 0.0,
        f"{metric_prefix}_retention_reference_top_only": 1.0 if retention_reference_top_only else 0.0,
        f"{metric_prefix}_retention_scoped": 1.0 if retention_scoped else 0.0,
        f"{metric_prefix}_top_action_retention_coef": float(top_action_retention_coef),
        f"{metric_prefix}_top_action_retention_margin": float(top_action_retention_margin),
        f"{metric_prefix}_top_action_retention_role_all": 1.0 if top_action_retention_role == "all" else 0.0,
        f"{metric_prefix}_top_action_retention_role_preferred": 1.0
        if top_action_retention_role == "preferred"
        else 0.0,
        f"{metric_prefix}_top_action_retention_role_rejected": 1.0 if top_action_retention_role == "rejected" else 0.0,
        f"{metric_prefix}_top_action_retention_reference_top_only": 1.0
        if top_action_retention_reference_top_only
        else 0.0,
        f"{metric_prefix}_top_action_retention_scoped": 1.0 if top_action_retention_scoped else 0.0,
    }
    metrics.update(retention_metrics)
    metrics.update(top_retention_metrics)
    tensors = {
        f"{metric_prefix}_margins": margin_tensor.detach(),
        f"{metric_prefix}_pair_weights": pair_weight_tensor.detach(),
    }
    tensors.update(retention_tensors)
    tensors.update(top_retention_tensors)
    return metrics, tensors


def empty_paired_outcome_preference_metrics(*, metric_prefix: str, aggregation: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_loss": 0.0,
        f"{metric_prefix}_pair_count": 0.0,
        f"{metric_prefix}_edge_count": 0.0,
        f"{metric_prefix}_candidate_pair_count": 0.0,
        f"{metric_prefix}_incomplete_pair_count": 0.0,
        f"{metric_prefix}_valid_rows": 0.0,
        f"{metric_prefix}_margin_mean": 0.0,
        f"{metric_prefix}_margin_min": 0.0,
        f"{metric_prefix}_satisfied_fraction": 0.0,
        f"{metric_prefix}_current_preferred_logp_mean": 0.0,
        f"{metric_prefix}_current_rejected_logp_mean": 0.0,
        f"{metric_prefix}_beta": 0.0,
        f"{metric_prefix}_aggregation_sum": 1.0 if aggregation == "sum" else 0.0,
        f"{metric_prefix}_aggregation_edge_mean": 1.0 if aggregation == "edge_mean" else 0.0,
        f"{metric_prefix}_group_balance": 0.0,
        f"{metric_prefix}_group_count": 0.0,
        f"{metric_prefix}_retention_coef": 0.0,
        f"{metric_prefix}_retention_margin": 0.0,
        f"{metric_prefix}_retention_role_all": 0.0,
        f"{metric_prefix}_retention_role_preferred": 0.0,
        f"{metric_prefix}_retention_role_rejected": 0.0,
        f"{metric_prefix}_retention_reference_top_only": 0.0,
        f"{metric_prefix}_retention_scoped": 0.0,
        f"{metric_prefix}_top_action_retention_coef": 0.0,
        f"{metric_prefix}_top_action_retention_margin": 0.0,
        f"{metric_prefix}_top_action_retention_role_all": 0.0,
        f"{metric_prefix}_top_action_retention_role_preferred": 0.0,
        f"{metric_prefix}_top_action_retention_role_rejected": 0.0,
        f"{metric_prefix}_top_action_retention_reference_top_only": 0.0,
        f"{metric_prefix}_top_action_retention_scoped": 0.0,
        **empty_retention_metrics(metric_prefix=metric_prefix),
        **empty_top_action_retention_metrics(metric_prefix=metric_prefix),
    }


__all__ = [
    "empty_paired_outcome_preference_metrics",
    "paired_outcome_preference_metrics_and_tensors",
]
