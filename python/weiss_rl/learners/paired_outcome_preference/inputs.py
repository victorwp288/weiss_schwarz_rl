"""Input normalization for paired-outcome preference losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.retention import finite_graph_zero


@dataclass(frozen=True)
class PairedOutcomePreferenceLossOptions:
    aggregation: str
    retention_role: str
    top_action_retention_role: str


@dataclass(frozen=True)
class PreparedPairedOutcomePreferenceInputs:
    options: PairedOutcomePreferenceLossOptions
    device: torch.device
    dtype: torch.dtype
    zero: Tensor
    current: Tensor
    reference: Tensor
    best_non_target: Tensor | None
    reference_best_non_target: Tensor | None
    pair_ids: Tensor
    roles: Tensor
    group_ids: Tensor | None
    pair_weight_rows: Tensor | None
    mask: Tensor
    retention_scope: Tensor | None
    top_retention_scope: Tensor | None
    valid: Tensor
    valid_row_count: int
    unique_pair_ids: Tensor


def prepare_paired_outcome_preference_loss_inputs(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    current_best_non_target_logp: Tensor | None,
    reference_best_non_target_logp: Tensor | None,
    preference_pair_ids: Tensor,
    preference_role: Tensor,
    preference_group_ids: Tensor | None,
    preference_pair_weights: Tensor | None,
    loss_mask: Tensor,
    aggregation: str,
    group_balance: bool,
    retention_coef: float,
    retention_margin: float,
    retention_role: str,
    retention_scope_mask: Tensor | None,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    top_action_retention_role: str,
    top_action_retention_scope_mask: Tensor | None,
) -> PreparedPairedOutcomePreferenceInputs:
    _validate_shapes(
        current_action_logp=current_action_logp,
        reference_action_logp=reference_action_logp,
        current_best_non_target_logp=current_best_non_target_logp,
        reference_best_non_target_logp=reference_best_non_target_logp,
        preference_pair_ids=preference_pair_ids,
        preference_role=preference_role,
        preference_group_ids=preference_group_ids,
        preference_pair_weights=preference_pair_weights,
        loss_mask=loss_mask,
        retention_scope_mask=retention_scope_mask,
        top_action_retention_scope_mask=top_action_retention_scope_mask,
    )
    options = _normalize_options(
        aggregation=aggregation,
        retention_role=retention_role,
        top_action_retention_role=top_action_retention_role,
    )
    _validate_loss_options(
        group_balance=group_balance,
        preference_group_ids=preference_group_ids,
        retention_coef=retention_coef,
        retention_margin=retention_margin,
        top_action_retention_coef=top_action_retention_coef,
        top_action_retention_margin=top_action_retention_margin,
    )

    device = current_action_logp.device
    dtype = current_action_logp.dtype
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
    _validate_pair_weights(pair_weight_rows=pair_weight_rows, valid=valid)
    valid_row_count = int(valid.sum().detach().cpu().item())
    unique_pair_ids = (
        torch.unique(pair_ids[valid], sorted=True) if valid_row_count > 0 else torch.empty(0, device=device)
    )
    return PreparedPairedOutcomePreferenceInputs(
        options=options,
        device=device,
        dtype=dtype,
        zero=finite_graph_zero(current_action_logp),
        current=current,
        reference=reference,
        best_non_target=best_non_target,
        reference_best_non_target=reference_best_non_target,
        pair_ids=pair_ids,
        roles=roles,
        group_ids=group_ids,
        pair_weight_rows=pair_weight_rows,
        mask=mask,
        retention_scope=retention_scope,
        top_retention_scope=top_retention_scope,
        valid=valid,
        valid_row_count=valid_row_count,
        unique_pair_ids=unique_pair_ids,
    )


def _validate_shapes(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    current_best_non_target_logp: Tensor | None,
    reference_best_non_target_logp: Tensor | None,
    preference_pair_ids: Tensor,
    preference_role: Tensor,
    preference_group_ids: Tensor | None,
    preference_pair_weights: Tensor | None,
    loss_mask: Tensor,
    retention_scope_mask: Tensor | None,
    top_action_retention_scope_mask: Tensor | None,
) -> None:
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
    if loss_mask.shape != current_action_logp.shape:
        raise ValueError("loss_mask must match logp shape")


def _normalize_options(
    *,
    aggregation: str,
    retention_role: str,
    top_action_retention_role: str,
) -> PairedOutcomePreferenceLossOptions:
    normalized_aggregation = str(aggregation).strip().lower()
    if normalized_aggregation not in {"mean", "sum", "edge_mean"}:
        raise ValueError("paired outcome preference aggregation must be one of: mean, sum, edge_mean")
    normalized_retention_role = str(retention_role).strip().lower()
    if normalized_retention_role not in {"all", "preferred", "rejected"}:
        raise ValueError("paired outcome preference retention_role must be one of: all, preferred, rejected")
    normalized_top_retention_role = str(top_action_retention_role).strip().lower()
    if normalized_top_retention_role not in {"all", "preferred", "rejected"}:
        raise ValueError("paired outcome preference top_action_retention_role must be one of: all, preferred, rejected")
    return PairedOutcomePreferenceLossOptions(
        aggregation=normalized_aggregation,
        retention_role=normalized_retention_role,
        top_action_retention_role=normalized_top_retention_role,
    )


def _validate_loss_options(
    *,
    group_balance: bool,
    preference_group_ids: Tensor | None,
    retention_coef: float,
    retention_margin: float,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
) -> None:
    if group_balance and preference_group_ids is None:
        raise ValueError("preference_group_ids are required when group_balance is enabled")
    if float(retention_coef) < 0.0:
        raise ValueError("paired outcome preference retention_coef must be >= 0")
    if float(retention_margin) < 0.0:
        raise ValueError("paired outcome preference retention_margin must be >= 0")
    if float(top_action_retention_coef) < 0.0:
        raise ValueError("paired outcome preference top_action_retention_coef must be >= 0")
    if float(top_action_retention_margin) < 0.0:
        raise ValueError("paired outcome preference top_action_retention_margin must be >= 0")


def _validate_pair_weights(*, pair_weight_rows: Tensor | None, valid: Tensor) -> None:
    if pair_weight_rows is None:
        return
    valid_weights = pair_weight_rows[valid]
    if bool((~torch.isfinite(valid_weights)).any().detach().cpu().item()):
        raise ValueError("preference_pair_weights must be finite on valid rows")
    if bool((valid_weights <= 0.0).any().detach().cpu().item()):
        raise ValueError("preference_pair_weights must be positive on valid rows")


__all__ = [
    "PairedOutcomePreferenceLossOptions",
    "PreparedPairedOutcomePreferenceInputs",
    "prepare_paired_outcome_preference_loss_inputs",
]
