"""Trajectory/span preference loss for paired outcome repair."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.inputs import (
    prepare_paired_outcome_preference_loss_inputs,
)
from weiss_rl.learners.paired_outcome_preference.metrics import (
    empty_paired_outcome_preference_metrics,
    paired_outcome_preference_metrics_and_tensors,
)
from weiss_rl.learners.paired_outcome_preference.pairs import (
    balanced_pair_loss,
    preference_pair_components,
    weighted_pair_loss,
)
from weiss_rl.learners.paired_outcome_preference.retention import (
    preference_retention_loss_and_metrics,
    preference_top_action_retention_loss_and_metrics,
)


def paired_outcome_preference_loss(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    current_best_non_target_logp: Tensor | None = None,
    reference_best_non_target_logp: Tensor | None = None,
    preference_pair_ids: Tensor,
    preference_role: Tensor,
    preference_group_ids: Tensor | None = None,
    preference_pair_weights: Tensor | None = None,
    loss_mask: Tensor,
    beta: float = 0.1,
    aggregation: str = "mean",
    group_balance: bool = False,
    retention_coef: float = 0.0,
    retention_margin: float = 0.0,
    retention_role: str = "preferred",
    retention_reference_top_only: bool = False,
    retention_scope_mask: Tensor | None = None,
    top_action_retention_coef: float = 0.0,
    top_action_retention_margin: float = 0.0,
    top_action_retention_role: str = "all",
    top_action_retention_reference_top_only: bool = False,
    top_action_retention_scope_mask: Tensor | None = None,
    metric_prefix: str = "paired_outcome_preference",
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Prefer one trajectory/span over another within exact paired-outcome groups.

    ``preference_role`` uses ``1`` for the preferred trajectory/span and ``0``
    for the rejected trajectory/span. Rows with a negative pair id, invalid role,
    or false ``loss_mask`` are ignored. The loss is DPO-style:

    ``-logsigmoid(beta * ((cur_pref - ref_pref) - (cur_rej - ref_rej)))``.
    """

    prepared = prepare_paired_outcome_preference_loss_inputs(
        current_action_logp=current_action_logp,
        reference_action_logp=reference_action_logp,
        current_best_non_target_logp=current_best_non_target_logp,
        reference_best_non_target_logp=reference_best_non_target_logp,
        preference_pair_ids=preference_pair_ids,
        preference_role=preference_role,
        preference_group_ids=preference_group_ids,
        preference_pair_weights=preference_pair_weights,
        loss_mask=loss_mask,
        aggregation=aggregation,
        group_balance=group_balance,
        retention_coef=retention_coef,
        retention_margin=retention_margin,
        retention_role=retention_role,
        retention_scope_mask=retention_scope_mask,
        top_action_retention_coef=top_action_retention_coef,
        top_action_retention_margin=top_action_retention_margin,
        top_action_retention_role=top_action_retention_role,
        top_action_retention_scope_mask=top_action_retention_scope_mask,
    )
    dtype = prepared.dtype
    zero = prepared.zero
    current = prepared.current
    reference = prepared.reference
    best_non_target = prepared.best_non_target
    reference_best_non_target = prepared.reference_best_non_target
    roles = prepared.roles
    valid = prepared.valid
    valid_row_count = prepared.valid_row_count
    normalized_aggregation = prepared.options.aggregation
    normalized_retention_role = prepared.options.retention_role
    normalized_top_retention_role = prepared.options.top_action_retention_role
    if valid_row_count <= 0:
        metrics = empty_paired_outcome_preference_metrics(
            metric_prefix=metric_prefix,
            aggregation=normalized_aggregation,
        )
        return zero, metrics, {}

    pair_components = preference_pair_components(
        current_action_logp=current_action_logp,
        reference_action_logp=reference_action_logp,
        current=current,
        reference=reference,
        pair_ids=prepared.pair_ids,
        roles=roles,
        valid=valid,
        group_ids=prepared.group_ids,
        pair_weight_rows=prepared.pair_weight_rows,
        preference_pair_ids=preference_pair_ids,
        preference_role=preference_role,
        preference_group_ids=preference_group_ids,
        preference_pair_weights=preference_pair_weights,
        unique_pair_ids=prepared.unique_pair_ids,
        aggregation=normalized_aggregation,
        beta=float(beta),
        dtype=dtype,
    )

    if not pair_components.margins:
        metrics = empty_paired_outcome_preference_metrics(
            metric_prefix=metric_prefix,
            aggregation=normalized_aggregation,
        )
        metrics[f"{metric_prefix}_valid_rows"] = float(valid_row_count)
        metrics[f"{metric_prefix}_incomplete_pair_count"] = float(pair_components.incomplete_pair_count)
        metrics[f"{metric_prefix}_candidate_pair_count"] = float(prepared.unique_pair_ids.numel())
        return zero, metrics, {}

    margin_tensor = torch.stack(pair_components.margins).to(dtype=dtype)
    pair_loss_tensor = torch.stack(pair_components.pair_losses).to(dtype=dtype)
    pair_weight_tensor = torch.stack(pair_components.pair_weights).to(device=prepared.device, dtype=dtype)
    loss = (
        balanced_pair_loss(pair_loss_tensor, pair_components.pair_group_ids, pair_weight_tensor)
        if group_balance
        else weighted_pair_loss(pair_loss_tensor, pair_weight_tensor)
    )
    loss = loss.to(dtype=dtype)
    retention_loss, retention_metrics, retention_tensors = preference_retention_loss_and_metrics(
        current=current,
        reference=reference,
        roles=roles,
        valid=valid,
        role=normalized_retention_role,
        scope_mask=prepared.retention_scope,
        margin=float(retention_margin),
        reference_best_non_target=reference_best_non_target,
        reference_top_only=bool(retention_reference_top_only),
        dtype=dtype,
        metric_prefix=metric_prefix,
    )
    if float(retention_coef) != 0.0:
        loss = loss + (retention_loss * float(retention_coef))
    top_retention_loss, top_retention_metrics, top_retention_tensors = preference_top_action_retention_loss_and_metrics(
        current=current,
        best_non_target=best_non_target,
        roles=roles,
        valid=valid,
        role=normalized_top_retention_role,
        scope_mask=prepared.top_retention_scope,
        margin=float(top_action_retention_margin),
        reference=reference,
        reference_best_non_target=reference_best_non_target,
        reference_top_only=bool(top_action_retention_reference_top_only),
        dtype=dtype,
        metric_prefix=metric_prefix,
    )
    if float(top_action_retention_coef) != 0.0:
        loss = loss + (top_retention_loss * float(top_action_retention_coef))
    metrics, tensors = paired_outcome_preference_metrics_and_tensors(
        loss=loss,
        pair_components=pair_components,
        margin_tensor=margin_tensor,
        pair_weight_tensor=pair_weight_tensor,
        candidate_pair_count=int(prepared.unique_pair_ids.numel()),
        valid_row_count=valid_row_count,
        metric_prefix=metric_prefix,
        beta=float(beta),
        aggregation=normalized_aggregation,
        group_balance=bool(group_balance),
        pair_weighted=preference_pair_weights is not None,
        retention_coef=float(retention_coef),
        retention_margin=float(retention_margin),
        retention_role=normalized_retention_role,
        retention_reference_top_only=bool(retention_reference_top_only),
        retention_scoped=retention_scope_mask is not None,
        top_action_retention_coef=float(top_action_retention_coef),
        top_action_retention_margin=float(top_action_retention_margin),
        top_action_retention_role=normalized_top_retention_role,
        top_action_retention_reference_top_only=bool(top_action_retention_reference_top_only),
        top_action_retention_scoped=top_action_retention_scope_mask is not None,
        retention_metrics=retention_metrics,
        top_retention_metrics=top_retention_metrics,
        retention_tensors=retention_tensors,
        top_retention_tensors=top_retention_tensors,
    )
    return loss, metrics, tensors


__all__ = ["paired_outcome_preference_loss"]
