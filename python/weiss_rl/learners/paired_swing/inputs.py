"""Input validation and row preparation for paired-swing losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairedSwingLossOptions:
    loss_scope: str
    compare_to: str


@dataclass(frozen=True)
class PreparedPairedSwingInputs:
    options: PairedSwingLossOptions
    zero: Tensor
    flat_loss_mask: Tensor
    flat_positive_actions: Tensor
    flat_negative_actions: Tensor
    flat_negative_valid: Tensor
    active_rows: Tensor
    raw_weight_total: float
    train_weight_total: float
    candidate_metrics: dict[str, float]


def prepare_paired_swing_loss_inputs(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor | None,
    positive_actions: Tensor,
    negative_actions: Tensor,
    negative_valid: Tensor,
    loss_mask: Tensor,
    loss_scope: str,
    compare_to: str,
    margin_retention_coef: float,
    margin_retention_margin: float,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    metric_prefix: str,
) -> PreparedPairedSwingInputs:
    _validate_shapes(
        packed_logits=packed_logits,
        reference_packed_logits=reference_packed_logits,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=negative_valid,
        loss_mask=loss_mask,
    )
    options = _normalize_options(loss_scope=loss_scope, compare_to=compare_to)
    _validate_retention_options(
        margin_retention_coef=margin_retention_coef,
        margin_retention_margin=margin_retention_margin,
        top_action_retention_coef=top_action_retention_coef,
        top_action_retention_margin=top_action_retention_margin,
    )

    flat_loss_mask = loss_mask.reshape(-1).to(device=packed_logits.device, dtype=torch.float32)
    flat_positive_actions = positive_actions.reshape(-1).to(device=packed_logits.device, dtype=torch.long)
    flat_negative_actions = negative_actions.reshape(-1).to(device=packed_logits.device, dtype=torch.long)
    flat_negative_valid = negative_valid.reshape(-1).to(device=packed_logits.device, dtype=torch.bool)
    active_rows = (
        (flat_loss_mask > 0.0)
        & flat_negative_valid
        & (flat_positive_actions >= 0)
        & (flat_negative_actions >= 0)
        & (flat_positive_actions != flat_negative_actions)
    )
    raw_weight = flat_loss_mask[active_rows]
    raw_weight_total = float(raw_weight.sum().item()) if bool(active_rows.any().item()) else 0.0
    train_weight_total = float(flat_loss_mask.sum().item())
    candidate_metrics = {
        f"{metric_prefix}_candidate_rows": float(active_rows.sum().item()),
        f"{metric_prefix}_distinct_fraction": (
            0.0 if train_weight_total <= 0.0 else raw_weight_total / max(train_weight_total, 1.0e-8)
        ),
    }
    return PreparedPairedSwingInputs(
        options=options,
        zero=packed_logits.sum() * 0.0,
        flat_loss_mask=flat_loss_mask,
        flat_positive_actions=flat_positive_actions,
        flat_negative_actions=flat_negative_actions,
        flat_negative_valid=flat_negative_valid,
        active_rows=active_rows,
        raw_weight_total=raw_weight_total,
        train_weight_total=train_weight_total,
        candidate_metrics=candidate_metrics,
    )


def _validate_shapes(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor | None,
    positive_actions: Tensor,
    negative_actions: Tensor,
    negative_valid: Tensor,
    loss_mask: Tensor,
) -> None:
    if positive_actions.shape != negative_actions.shape:
        raise ValueError("positive_actions and negative_actions must have the same shape")
    if positive_actions.shape != loss_mask.shape:
        raise ValueError("paired-swing action tensors must match loss_mask shape")
    if negative_valid.shape != loss_mask.shape:
        raise ValueError("negative_valid must match loss_mask shape")
    if reference_packed_logits is not None and reference_packed_logits.shape != packed_logits.shape:
        raise ValueError("reference_packed_logits must match packed_logits shape")


def _normalize_options(*, loss_scope: str, compare_to: str) -> PairedSwingLossOptions:
    normalized_scope = str(loss_scope).strip().lower()
    if normalized_scope not in {"row", "episode_mean", "label_mean"}:
        raise ValueError("paired_swing loss_scope must be one of: row, episode_mean, label_mean")
    normalized_compare_to = str(compare_to).strip().lower()
    if normalized_compare_to not in {"negative", "top_other"}:
        raise ValueError("paired_swing compare_to must be one of: negative, top_other")
    return PairedSwingLossOptions(loss_scope=normalized_scope, compare_to=normalized_compare_to)


def _validate_retention_options(
    *,
    margin_retention_coef: float,
    margin_retention_margin: float,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
) -> None:
    if float(margin_retention_coef) < 0.0:
        raise ValueError("paired_swing margin_retention_coef must be >= 0")
    if float(margin_retention_margin) < 0.0:
        raise ValueError("paired_swing margin_retention_margin must be >= 0")
    if float(top_action_retention_coef) < 0.0:
        raise ValueError("paired_swing top_action_retention_coef must be >= 0")
    if float(top_action_retention_margin) < 0.0:
        raise ValueError("paired_swing top_action_retention_margin must be >= 0")


__all__ = [
    "PairedSwingLossOptions",
    "PreparedPairedSwingInputs",
    "prepare_paired_swing_loss_inputs",
]
