"""Paired-swing contrastive losses for replay repair batches."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.paired_swing.comparison import paired_swing_margin_comparison_rows
from weiss_rl.learners.paired_swing.inputs import prepare_paired_swing_loss_inputs
from weiss_rl.learners.paired_swing.metrics import (
    paired_swing_final_metrics,
    paired_swing_no_active_metrics,
    paired_swing_no_supported_metrics,
    paired_swing_output_tensors,
    paired_swing_supported_rows,
)
from weiss_rl.learners.paired_swing.retention import (
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
    paired_swing_margin_retention_loss_and_metrics,
    paired_swing_top_action_retention_loss_and_metrics,
)
from weiss_rl.learners.paired_swing.scope import paired_swing_scoped_margin_loss


def packed_paired_swing_margin_loss(
    *,
    packed_logits: Tensor,
    reference_packed_logits: Tensor | None = None,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    positive_actions: Tensor,
    negative_actions: Tensor,
    negative_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    pass_action_id: int | None,
    loss_scope: str = "row",
    compare_to: str = "negative",
    group_ids: Tensor | None = None,
    margin_retention_coef: float = 0.0,
    margin_retention_margin: float = 0.0,
    top_action_retention_coef: float = 0.0,
    top_action_retention_margin: float = 0.0,
    metric_prefix: str = "paired_swing",
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Require positive swing actions to outrank paired negative actions.

    ``positive_actions`` and ``negative_actions`` are time-major tensors aligned
    to the replay batch. The function is intentionally action-pair based rather
    than teacher-BC based: rows where the two actions match do not train.
    """

    prepared = prepare_paired_swing_loss_inputs(
        packed_logits=packed_logits,
        reference_packed_logits=reference_packed_logits,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=negative_valid,
        loss_mask=loss_mask,
        loss_scope=loss_scope,
        compare_to=compare_to,
        margin_retention_coef=margin_retention_coef,
        margin_retention_margin=margin_retention_margin,
        top_action_retention_coef=top_action_retention_coef,
        top_action_retention_margin=top_action_retention_margin,
        metric_prefix=metric_prefix,
    )
    zero = prepared.zero
    flat_loss_mask = prepared.flat_loss_mask
    flat_positive_actions = prepared.flat_positive_actions
    flat_negative_actions = prepared.flat_negative_actions
    active_rows = prepared.active_rows
    raw_weight_total = prepared.raw_weight_total
    normalized_scope = prepared.options.loss_scope
    normalized_compare_to = prepared.options.compare_to
    if not bool(active_rows.any().item()) or raw_weight_total <= 0.0:
        metrics = paired_swing_no_active_metrics(
            candidate_metrics=prepared.candidate_metrics,
            metric_prefix=metric_prefix,
        )
        return zero, metrics, {}

    comparison = paired_swing_margin_comparison_rows(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        flat_positive_actions=flat_positive_actions,
        flat_negative_actions=flat_negative_actions,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        active_rows=active_rows,
        pass_action_id=pass_action_id,
        compare_to=normalized_compare_to,
    )
    margin_by_row = comparison.margin_by_row
    supported = comparison.supported
    positive_logp_by_row = comparison.positive_logp_by_row
    negative_logp_by_row = comparison.negative_logp_by_row
    supported_rows = paired_swing_supported_rows(
        margin_by_row=margin_by_row,
        positive_logp_by_row=positive_logp_by_row,
        negative_logp_by_row=negative_logp_by_row,
        supported=supported,
        flat_loss_mask=flat_loss_mask,
        raw_weight_total=raw_weight_total,
        packed_logits=packed_logits,
        candidate_metrics=prepared.candidate_metrics,
        metric_prefix=metric_prefix,
    )
    if not bool(supported.any().item()) or not supported_rows.has_weight:
        metrics = paired_swing_no_supported_metrics(
            row_metrics=supported_rows.metrics,
            metric_prefix=metric_prefix,
        )
        return zero, metrics, {}

    if normalized_scope == "label_mean":
        if group_ids is None:
            raise ValueError("paired_swing loss_scope label_mean requires group_ids")
        if group_ids.shape != loss_mask.shape:
            raise ValueError("paired-swing group_ids must match loss_mask shape")
        scope_group_ids = group_ids.reshape(-1).to(device=packed_logits.device, dtype=torch.long)[supported]
    else:
        scope_group_ids = None
    loss, margin_mean, satisfied_fraction, scope_metrics = paired_swing_scoped_margin_loss(
        margins=supported_rows.margins,
        supported_weight=supported_rows.supported_weight,
        supported=supported,
        positive_actions=positive_actions,
        group_ids=scope_group_ids,
        normalized_scope=normalized_scope,
        margin=float(margin),
    )
    retention_loss, retention_metrics, retention_tensors = paired_swing_margin_retention_loss_and_metrics(
        current_margin_by_row=margin_by_row,
        reference_packed_logits=reference_packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        flat_positive_actions=flat_positive_actions,
        flat_negative_actions=flat_negative_actions,
        active_rows=active_rows,
        supported=supported,
        supported_weight=flat_loss_mask,
        pass_action_id=pass_action_id,
        compare_to=normalized_compare_to,
        retention_margin=float(margin_retention_margin),
        metric_prefix=metric_prefix,
    )
    if float(margin_retention_coef) != 0.0:
        loss = loss + (retention_loss * float(margin_retention_coef))
    top_retention_loss, top_retention_metrics, top_retention_tensors = (
        paired_swing_top_action_retention_loss_and_metrics(
            packed_logits=packed_logits,
            reference_packed_logits=reference_packed_logits,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            supported=supported,
            supported_weight=flat_loss_mask,
            retention_margin=float(top_action_retention_margin),
            metric_prefix=metric_prefix,
        )
    )
    if float(top_action_retention_coef) != 0.0:
        loss = loss + (top_retention_loss * float(top_action_retention_coef))
    metrics = paired_swing_final_metrics(
        row_metrics=supported_rows.metrics,
        loss=loss,
        margin_mean=margin_mean,
        satisfied_fraction=satisfied_fraction,
        normalized_scope=normalized_scope,
        normalized_compare_to=normalized_compare_to,
        margin_retention_coef=margin_retention_coef,
        margin_retention_margin=margin_retention_margin,
        top_action_retention_coef=top_action_retention_coef,
        top_action_retention_margin=top_action_retention_margin,
        positive_metric_logp=supported_rows.positive_metric_logp,
        negative_metric_logp=supported_rows.negative_metric_logp,
        supported_weight=supported_rows.supported_weight,
        scope_metrics=scope_metrics,
        retention_metrics=retention_metrics,
        top_retention_metrics=top_retention_metrics,
        metric_prefix=metric_prefix,
    )
    tensors = paired_swing_output_tensors(
        margins=supported_rows.margins,
        retention_tensors=retention_tensors,
        top_retention_tensors=top_retention_tensors,
    )
    return loss, metrics, tensors


__all__ = [
    "packed_paired_swing_margin_loss",
    "packed_target_action_retention_loss",
    "packed_top_action_retention_loss",
]
