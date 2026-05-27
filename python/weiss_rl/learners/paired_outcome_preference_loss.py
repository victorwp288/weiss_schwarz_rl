"""Trajectory/span preference loss for paired outcome repair."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


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

    if current_action_logp.shape != reference_action_logp.shape:
        raise ValueError("current_action_logp and reference_action_logp must have the same shape")
    if current_best_non_target_logp is not None and current_best_non_target_logp.shape != current_action_logp.shape:
        raise ValueError("current_best_non_target_logp must match logp shape")
    if reference_best_non_target_logp is not None and reference_best_non_target_logp.shape != current_action_logp.shape:
        raise ValueError("reference_best_non_target_logp must match logp shape")
    if preference_pair_ids.shape != current_action_logp.shape:
        raise ValueError("preference_pair_ids must match logp shape")
    if preference_role.shape != current_action_logp.shape:
        raise ValueError("preference_role must match logp shape")
    if preference_group_ids is not None and preference_group_ids.shape != current_action_logp.shape:
        raise ValueError("preference_group_ids must match logp shape")
    if preference_pair_weights is not None and preference_pair_weights.shape != current_action_logp.shape:
        raise ValueError("preference_pair_weights must match logp shape")
    if retention_scope_mask is not None and retention_scope_mask.shape != current_action_logp.shape:
        raise ValueError("retention_scope_mask must match logp shape")
    if (
        top_action_retention_scope_mask is not None
        and top_action_retention_scope_mask.shape != current_action_logp.shape
    ):
        raise ValueError("top_action_retention_scope_mask must match logp shape")
    if group_balance and preference_group_ids is None:
        raise ValueError("preference_group_ids are required when group_balance is enabled")
    if loss_mask.shape != current_action_logp.shape:
        raise ValueError("loss_mask must match logp shape")
    normalized_aggregation = str(aggregation).strip().lower()
    if normalized_aggregation not in {"mean", "sum", "edge_mean"}:
        raise ValueError("paired outcome preference aggregation must be one of: mean, sum, edge_mean")
    normalized_retention_role = str(retention_role).strip().lower()
    if normalized_retention_role not in {"all", "preferred", "rejected"}:
        raise ValueError("paired outcome preference retention_role must be one of: all, preferred, rejected")
    normalized_top_retention_role = str(top_action_retention_role).strip().lower()
    if normalized_top_retention_role not in {"all", "preferred", "rejected"}:
        raise ValueError("paired outcome preference top_action_retention_role must be one of: all, preferred, rejected")
    if float(retention_coef) < 0.0:
        raise ValueError("paired outcome preference retention_coef must be >= 0")
    if float(retention_margin) < 0.0:
        raise ValueError("paired outcome preference retention_margin must be >= 0")
    if float(top_action_retention_coef) < 0.0:
        raise ValueError("paired outcome preference top_action_retention_coef must be >= 0")
    if float(top_action_retention_margin) < 0.0:
        raise ValueError("paired outcome preference top_action_retention_margin must be >= 0")

    device = current_action_logp.device
    dtype = current_action_logp.dtype
    zero = _finite_graph_zero(current_action_logp)
    current = current_action_logp.reshape(-1)
    reference = reference_action_logp.reshape(-1).to(device=device, dtype=dtype)
    best_non_target = (
        None
        if current_best_non_target_logp is None
        else current_best_non_target_logp.reshape(-1).to(device=device, dtype=dtype)
    )
    reference_best_non_target = (
        None
        if reference_best_non_target_logp is None
        else reference_best_non_target_logp.reshape(-1).to(device=device, dtype=dtype)
    )
    pair_ids = preference_pair_ids.reshape(-1).to(device=device, dtype=torch.long)
    roles = preference_role.reshape(-1).to(device=device, dtype=torch.long)
    group_ids = (
        None if preference_group_ids is None else preference_group_ids.reshape(-1).to(device=device, dtype=torch.long)
    )
    pair_weight_rows = (
        None if preference_pair_weights is None else preference_pair_weights.reshape(-1).to(device=device, dtype=dtype)
    )
    mask = loss_mask.reshape(-1).to(device=device, dtype=torch.bool)
    retention_scope = (
        None if retention_scope_mask is None else retention_scope_mask.reshape(-1).to(device=device, dtype=torch.bool)
    )
    top_retention_scope = (
        None
        if top_action_retention_scope_mask is None
        else top_action_retention_scope_mask.reshape(-1).to(device=device, dtype=torch.bool)
    )
    valid = mask & (pair_ids >= 0) & ((roles == 0) | (roles == 1)) & torch.isfinite(current) & torch.isfinite(reference)
    if pair_weight_rows is not None:
        valid_weights = pair_weight_rows[valid]
        if bool((~torch.isfinite(valid_weights)).any().detach().cpu().item()):
            raise ValueError("preference_pair_weights must be finite on valid rows")
        if bool((valid_weights <= 0.0).any().detach().cpu().item()):
            raise ValueError("preference_pair_weights must be positive on valid rows")
    valid_row_count = int(valid.sum().detach().cpu().item())
    if valid_row_count <= 0:
        metrics = _empty_metrics(metric_prefix=metric_prefix, aggregation=normalized_aggregation)
        return zero, metrics, {}

    unique_pair_ids = torch.unique(pair_ids[valid], sorted=True)
    if normalized_aggregation == "edge_mean":
        (
            margins,
            pair_losses,
            pair_weights,
            pair_group_ids,
            current_pref_values,
            current_rej_values,
            incomplete_pair_count,
        ) = _edge_mean_components(
            current_action_logp=current_action_logp,
            reference_action_logp=reference_action_logp,
            pair_ids=preference_pair_ids,
            roles=preference_role,
            valid=valid.reshape(current_action_logp.shape),
            group_ids=None if preference_group_ids is None else preference_group_ids,
            pair_weight_rows=None if preference_pair_weights is None else preference_pair_weights,
            unique_pair_ids=unique_pair_ids,
            beta=float(beta),
            dtype=dtype,
        )
    else:
        margins = []
        pair_losses = []
        pair_weights = []
        pair_group_ids = []
        current_pref_values = []
        current_rej_values = []
        incomplete_pair_count = 0
        for pair_id in unique_pair_ids:
            pair_mask = valid & (pair_ids == pair_id)
            preferred_mask = pair_mask & (roles == 1)
            rejected_mask = pair_mask & (roles == 0)
            if not bool(preferred_mask.any().item()) or not bool(rejected_mask.any().item()):
                incomplete_pair_count += 1
                continue
            cur_pref = _aggregate(current[preferred_mask], normalized_aggregation)
            ref_pref = _aggregate(reference[preferred_mask], normalized_aggregation)
            cur_rej = _aggregate(current[rejected_mask], normalized_aggregation)
            ref_rej = _aggregate(reference[rejected_mask], normalized_aggregation)
            margin = (cur_pref - ref_pref) - (cur_rej - ref_rej)
            margins.append(margin)
            pair_losses.append(-F.logsigmoid(margin * float(beta)))
            if pair_weight_rows is None:
                pair_weights.append(torch.ones((), device=device, dtype=dtype))
            else:
                pair_weights.append(pair_weight_rows[pair_mask].mean().detach().to(device=device, dtype=dtype))
            if group_ids is not None:
                group_values = group_ids[pair_mask]
                pair_group_ids.append(int(group_values[0].detach().cpu().item()) if group_values.numel() else -1)
            current_pref_values.append(cur_pref)
            current_rej_values.append(cur_rej)

    if not margins:
        metrics = _empty_metrics(metric_prefix=metric_prefix, aggregation=normalized_aggregation)
        metrics[f"{metric_prefix}_valid_rows"] = float(valid_row_count)
        metrics[f"{metric_prefix}_incomplete_pair_count"] = float(incomplete_pair_count)
        metrics[f"{metric_prefix}_candidate_pair_count"] = float(unique_pair_ids.numel())
        return zero, metrics, {}

    margin_tensor = torch.stack(margins).to(dtype=dtype)
    pair_loss_tensor = torch.stack(pair_losses).to(dtype=dtype)
    pair_weight_tensor = torch.stack(pair_weights).to(device=device, dtype=dtype)
    loss = (
        _balanced_pair_loss(pair_loss_tensor, pair_group_ids, pair_weight_tensor)
        if group_balance
        else _weighted_pair_loss(pair_loss_tensor, pair_weight_tensor)
    )
    loss = loss.to(dtype=dtype)
    retention_loss, retention_metrics, retention_tensors = _retention_loss_and_metrics(
        current=current,
        reference=reference,
        roles=roles,
        valid=valid,
        role=normalized_retention_role,
        scope_mask=retention_scope,
        margin=float(retention_margin),
        reference_best_non_target=reference_best_non_target,
        reference_top_only=bool(retention_reference_top_only),
        dtype=dtype,
        metric_prefix=metric_prefix,
    )
    if float(retention_coef) != 0.0:
        loss = loss + (retention_loss * float(retention_coef))
    top_retention_loss, top_retention_metrics, top_retention_tensors = _top_action_retention_loss_and_metrics(
        current=current,
        best_non_target=best_non_target,
        roles=roles,
        valid=valid,
        role=normalized_top_retention_role,
        scope_mask=top_retention_scope,
        margin=float(top_action_retention_margin),
        reference=reference,
        reference_best_non_target=reference_best_non_target,
        reference_top_only=bool(top_action_retention_reference_top_only),
        dtype=dtype,
        metric_prefix=metric_prefix,
    )
    if float(top_action_retention_coef) != 0.0:
        loss = loss + (top_retention_loss * float(top_action_retention_coef))
    satisfied = (margin_tensor > 0.0).to(dtype=dtype)
    current_pref_tensor = torch.stack(current_pref_values).to(dtype=dtype)
    current_rej_tensor = torch.stack(current_rej_values).to(dtype=dtype)
    metrics = {
        f"{metric_prefix}_loss": float(loss.detach().item()),
        f"{metric_prefix}_pair_count": float(margin_tensor.numel()),
        f"{metric_prefix}_edge_count": float(margin_tensor.numel()) if normalized_aggregation == "edge_mean" else 0.0,
        f"{metric_prefix}_candidate_pair_count": float(unique_pair_ids.numel()),
        f"{metric_prefix}_incomplete_pair_count": float(incomplete_pair_count),
        f"{metric_prefix}_valid_rows": float(valid_row_count),
        f"{metric_prefix}_margin_mean": float(margin_tensor.mean().detach().item()),
        f"{metric_prefix}_margin_min": float(margin_tensor.min().detach().item()),
        f"{metric_prefix}_satisfied_fraction": float(satisfied.mean().detach().item()),
        f"{metric_prefix}_current_preferred_logp_mean": float(current_pref_tensor.mean().detach().item()),
        f"{metric_prefix}_current_rejected_logp_mean": float(current_rej_tensor.mean().detach().item()),
        f"{metric_prefix}_beta": float(beta),
        f"{metric_prefix}_aggregation_sum": 1.0 if normalized_aggregation == "sum" else 0.0,
        f"{metric_prefix}_aggregation_edge_mean": 1.0 if normalized_aggregation == "edge_mean" else 0.0,
        f"{metric_prefix}_group_balance": 1.0 if group_balance else 0.0,
        f"{metric_prefix}_group_count": float(len(set(pair_group_ids))) if group_balance else 0.0,
        f"{metric_prefix}_pair_weighted": 1.0 if preference_pair_weights is not None else 0.0,
        f"{metric_prefix}_pair_weight_mean": float(pair_weight_tensor.mean().detach().item()),
        f"{metric_prefix}_pair_weight_min": float(pair_weight_tensor.min().detach().item()),
        f"{metric_prefix}_pair_weight_max": float(pair_weight_tensor.max().detach().item()),
        f"{metric_prefix}_pair_weight_nondefault_count": float(
            torch.count_nonzero(torch.abs(pair_weight_tensor - 1.0) > 1e-12).detach().item()
        ),
        f"{metric_prefix}_retention_coef": float(retention_coef),
        f"{metric_prefix}_retention_margin": float(retention_margin),
        f"{metric_prefix}_retention_role_all": 1.0 if normalized_retention_role == "all" else 0.0,
        f"{metric_prefix}_retention_role_preferred": 1.0 if normalized_retention_role == "preferred" else 0.0,
        f"{metric_prefix}_retention_role_rejected": 1.0 if normalized_retention_role == "rejected" else 0.0,
        f"{metric_prefix}_retention_reference_top_only": 1.0 if retention_reference_top_only else 0.0,
        f"{metric_prefix}_retention_scoped": 1.0 if retention_scope_mask is not None else 0.0,
        f"{metric_prefix}_top_action_retention_coef": float(top_action_retention_coef),
        f"{metric_prefix}_top_action_retention_margin": float(top_action_retention_margin),
        f"{metric_prefix}_top_action_retention_role_all": 1.0 if normalized_top_retention_role == "all" else 0.0,
        f"{metric_prefix}_top_action_retention_role_preferred": 1.0
        if normalized_top_retention_role == "preferred"
        else 0.0,
        f"{metric_prefix}_top_action_retention_role_rejected": 1.0
        if normalized_top_retention_role == "rejected"
        else 0.0,
        f"{metric_prefix}_top_action_retention_reference_top_only": 1.0
        if top_action_retention_reference_top_only
        else 0.0,
        f"{metric_prefix}_top_action_retention_scoped": 1.0 if top_action_retention_scope_mask is not None else 0.0,
    }
    metrics.update(retention_metrics)
    metrics.update(top_retention_metrics)
    tensors = {
        f"{metric_prefix}_margins": margin_tensor.detach(),
        f"{metric_prefix}_pair_weights": pair_weight_tensor.detach(),
    }
    tensors.update(retention_tensors)
    tensors.update(top_retention_tensors)
    return loss, metrics, tensors


def _aggregate(values: Tensor, aggregation: str) -> Tensor:
    if aggregation == "sum":
        return values.sum()
    return values.mean()


def _edge_mean_components(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    pair_ids: Tensor,
    roles: Tensor,
    valid: Tensor,
    group_ids: Tensor | None,
    pair_weight_rows: Tensor | None,
    unique_pair_ids: Tensor,
    beta: float,
    dtype: torch.dtype,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[int], list[Tensor], list[Tensor], int]:
    if current_action_logp.ndim != 2:
        raise ValueError("paired outcome preference edge_mean aggregation requires 2D time-major tensors")
    device = current_action_logp.device
    current = current_action_logp.to(device=device, dtype=dtype)
    reference = reference_action_logp.to(device=device, dtype=dtype)
    pair_ids_2d = pair_ids.to(device=device, dtype=torch.long)
    roles_2d = roles.to(device=device, dtype=torch.long)
    valid_2d = valid.to(device=device, dtype=torch.bool)
    group_ids_2d = None if group_ids is None else group_ids.to(device=device, dtype=torch.long)
    weights_2d = None if pair_weight_rows is None else pair_weight_rows.to(device=device, dtype=dtype)

    margins: list[Tensor] = []
    edge_losses: list[Tensor] = []
    edge_weights: list[Tensor] = []
    edge_group_ids: list[int] = []
    current_pref_values: list[Tensor] = []
    current_rej_values: list[Tensor] = []
    incomplete_pair_count = 0
    time_steps = int(current.shape[0])
    for pair_id in unique_pair_ids:
        pair_mask = valid_2d & (pair_ids_2d == pair_id)
        preferred_mask = pair_mask & (roles_2d == 1)
        rejected_mask = pair_mask & (roles_2d == 0)
        if not bool(preferred_mask.any().item()) or not bool(rejected_mask.any().item()):
            incomplete_pair_count += 1
            continue
        pair_group_id = -1
        if group_ids_2d is not None:
            group_values = group_ids_2d[pair_mask]
            pair_group_id = int(group_values[0].detach().cpu().item()) if group_values.numel() else -1
        edge_count = 0
        for step in range(time_steps):
            preferred_step = preferred_mask[step]
            rejected_step = rejected_mask[step]
            if not bool(preferred_step.any().item()) or not bool(rejected_step.any().item()):
                continue
            cur_pref = current[step][preferred_step].mean()
            ref_pref = reference[step][preferred_step].mean()
            cur_rej = current[step][rejected_step].mean()
            ref_rej = reference[step][rejected_step].mean()
            margin = (cur_pref - ref_pref) - (cur_rej - ref_rej)
            margins.append(margin)
            edge_losses.append(-F.logsigmoid(margin * beta))
            if weights_2d is None:
                edge_weights.append(torch.ones((), device=device, dtype=dtype))
            else:
                edge_mask = preferred_step | rejected_step
                edge_weights.append(weights_2d[step][edge_mask].mean().detach().to(device=device, dtype=dtype))
            edge_group_ids.append(pair_group_id)
            current_pref_values.append(cur_pref)
            current_rej_values.append(cur_rej)
            edge_count += 1
        if edge_count <= 0:
            incomplete_pair_count += 1
    return (
        margins,
        edge_losses,
        edge_weights,
        edge_group_ids,
        current_pref_values,
        current_rej_values,
        incomplete_pair_count,
    )


def _weighted_pair_loss(pair_losses: Tensor, pair_weights: Tensor) -> Tensor:
    if pair_losses.shape != pair_weights.shape:
        raise ValueError("pair_weights must have the same shape as pair_losses")
    weight_sum = pair_weights.sum()
    if bool((weight_sum <= 0.0).detach().cpu().item()):
        return pair_losses.mean()
    return (pair_losses * pair_weights).sum() / weight_sum


def _balanced_pair_loss(pair_losses: Tensor, pair_group_ids: list[int], pair_weights: Tensor) -> Tensor:
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
            group_losses.append(_weighted_pair_loss(pair_losses[group_mask], pair_weights[group_mask]))
    if not group_losses:
        return pair_losses.mean()
    return torch.stack(group_losses).mean()


def _retention_loss_and_metrics(
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
    zero = _finite_graph_zero(current)
    row_count = int(retention_mask.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, _empty_retention_metrics(metric_prefix=metric_prefix), {}
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


def _top_action_retention_loss_and_metrics(
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
    zero = _finite_graph_zero(current)
    if best_non_target is None:
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
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
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
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


def _empty_metrics(*, metric_prefix: str, aggregation: str) -> dict[str, float]:
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
        **_empty_retention_metrics(metric_prefix=metric_prefix),
        **_empty_top_action_retention_metrics(metric_prefix=metric_prefix),
    }


def _empty_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_retention_loss": 0.0,
        f"{metric_prefix}_retention_row_count": 0.0,
        f"{metric_prefix}_retention_violation_fraction": 0.0,
        f"{metric_prefix}_retention_logp_delta_mean": 0.0,
        f"{metric_prefix}_retention_logp_delta_min": 0.0,
    }


def _empty_top_action_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_top_action_retention_loss": 0.0,
        f"{metric_prefix}_top_action_retention_row_count": 0.0,
        f"{metric_prefix}_top_action_retention_violation_fraction": 0.0,
        f"{metric_prefix}_top_action_retention_gap_mean": 0.0,
        f"{metric_prefix}_top_action_retention_gap_min": 0.0,
    }


def _finite_graph_zero(values: Tensor) -> Tensor:
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return finite_values.sum() * 0.0


__all__ = ["paired_outcome_preference_loss"]
