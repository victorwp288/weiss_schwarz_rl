"""Candidate-level teacher-action margin losses."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.tensor_ops import segment_max, weighted_mean


def _margin_metrics(
    *,
    margins: Tensor,
    weights: Tensor,
    violations: Tensor,
    requested_margin: float,
    metric_prefix: str,
) -> tuple[Tensor, dict[str, float]]:
    loss = weighted_mean(violations, weights)
    weight_total = max(float(weights.sum().item()), 1.0)
    return loss, {
        f"{metric_prefix}_loss": float(loss.detach().item()),
        f"{metric_prefix}_mean": float(weighted_mean(margins, weights).detach().item()),
        f"{metric_prefix}_satisfied_fraction": float(
            (((margins >= float(requested_margin)).to(dtype=weights.dtype) * weights).sum().item()) / weight_total
        ),
    }


def packed_teacher_action_margin_loss(
    *,
    packed_view: PackedStructuredLegalView,
    teacher_action: Tensor,
    teacher_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Hinge loss requiring teacher action logits to beat legal competitors."""

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_action = teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    action_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_action >= 0)
    row_weights = flat_loss_mask[action_rows]
    if not bool(action_rows.any().item()) or float(row_weights.sum().item()) <= 0.0:
        return zero, {}, {}

    row_teacher_actions = flat_teacher_action.index_select(
        0,
        packed_view.row_indices.to(dtype=torch.long),
    )
    teacher_candidates = packed_view.action_ids.to(dtype=torch.long) == row_teacher_actions
    neg_inf = torch.full_like(packed_view.logits, -torch.inf)

    teacher_logits = segment_max(
        torch.where(teacher_candidates, packed_view.logits, neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    competitor_logits = segment_max(
        torch.where(teacher_candidates, neg_inf, packed_view.logits),
        packed_view.row_indices,
        packed_view.row_count,
    )
    supported = action_rows & torch.isfinite(teacher_logits) & torch.isfinite(competitor_logits)
    if float(row_weights.sum().item()) > 0.0:
        supported_fraction = float(
            flat_loss_mask[supported].sum().item() / max(float(row_weights.sum().item()), 1.0e-8)
        )
    else:
        supported_fraction = 0.0
    if not bool(supported.any().item()):
        return zero, {"teacher_action_margin_supported_fraction": supported_fraction}, {}

    supported_weights = flat_loss_mask[supported]
    margins = (teacher_logits[supported] - competitor_logits[supported]).to(dtype=value_dtype)
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss, metrics = _margin_metrics(
        margins=margins,
        weights=supported_weights,
        violations=violations,
        requested_margin=float(margin),
        metric_prefix="teacher_action_margin",
    )
    metrics["teacher_action_margin_supported_fraction"] = supported_fraction
    return loss.to(dtype=value_dtype), metrics, {"teacher_action_margins": margins.detach()}


def dense_teacher_action_margin_loss(
    *,
    logits: Tensor,
    legal_mask: Tensor,
    teacher_action: Tensor,
    teacher_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    zero: Tensor,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Dense-mask equivalent of the candidate-level teacher-action margin loss."""

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_action = teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_legal_mask = legal_mask.reshape(-1, legal_mask.shape[-1]).to(dtype=torch.bool)
    action_rows = flat_teacher_valid & (flat_teacher_action >= 0)
    action_row_indices = torch.nonzero(action_rows, as_tuple=False).squeeze(1)
    row_weights = flat_loss_mask[action_rows]
    if action_row_indices.numel() == 0 or float(row_weights.sum().item()) <= 0.0:
        return zero, {}, {}

    action_targets = flat_teacher_action[action_rows]
    in_range = (action_targets >= 0) & (action_targets < flat_logits.shape[-1])
    if not bool(in_range.any().item()):
        return zero, {"teacher_action_margin_supported_fraction": 0.0}, {}

    row_indices = action_row_indices[in_range]
    targets = action_targets[in_range]
    selected_logits = flat_logits[row_indices]
    selected_masks = flat_legal_mask[row_indices]
    selected_weights = flat_loss_mask[row_indices]
    teacher_legal = selected_masks.gather(1, targets.unsqueeze(1)).squeeze(1)
    competitor_logits = torch.where(selected_masks, selected_logits, torch.full_like(selected_logits, -torch.inf))
    competitor_logits = competitor_logits.scatter(1, targets.unsqueeze(1), -torch.inf)
    competitor_max = competitor_logits.max(dim=1).values
    supported = teacher_legal & torch.isfinite(competitor_max)
    supported_fraction = float(selected_weights[supported].sum().item() / max(float(row_weights.sum().item()), 1.0e-8))
    if not bool(supported.any().item()):
        return zero, {"teacher_action_margin_supported_fraction": supported_fraction}, {}

    target_logits = selected_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    margins = (target_logits[supported] - competitor_max[supported]).to(dtype=logits.dtype)
    supported_weights = selected_weights[supported]
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss, metrics = _margin_metrics(
        margins=margins,
        weights=supported_weights,
        violations=violations,
        requested_margin=float(margin),
        metric_prefix="teacher_action_margin",
    )
    metrics["teacher_action_margin_supported_fraction"] = supported_fraction
    return loss.to(dtype=logits.dtype), metrics, {"teacher_action_margins": margins.detach()}


def packed_teacher_same_family_action_margin_loss(
    *,
    packed_view: PackedStructuredLegalView,
    teacher_action: Tensor,
    teacher_family: Tensor,
    teacher_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Hinge loss requiring teacher actions to beat same-family competitors."""

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_action = teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    action_rows = (
        packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
    )
    row_weights = flat_loss_mask[action_rows]
    if not bool(action_rows.any().item()) or float(row_weights.sum().item()) <= 0.0:
        return zero, {}, {}

    candidate_rows = packed_view.row_indices.to(dtype=torch.long)
    row_teacher_actions = flat_teacher_action.index_select(0, candidate_rows)
    row_teacher_families = flat_teacher_family.index_select(0, candidate_rows)
    same_family_candidates = packed_view.family_ids.to(dtype=torch.long) == row_teacher_families
    teacher_candidates = same_family_candidates & (packed_view.action_ids.to(dtype=torch.long) == row_teacher_actions)
    same_family_competitors = same_family_candidates & ~teacher_candidates
    neg_inf = torch.full_like(packed_view.logits, -torch.inf)

    teacher_logits = segment_max(
        torch.where(teacher_candidates, packed_view.logits, neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    competitor_logits = segment_max(
        torch.where(same_family_competitors, packed_view.logits, neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    supported = action_rows & torch.isfinite(teacher_logits) & torch.isfinite(competitor_logits)
    supported_fraction = float(flat_loss_mask[supported].sum().item() / max(float(row_weights.sum().item()), 1.0e-8))
    if not bool(supported.any().item()):
        return zero, {"teacher_same_family_action_margin_supported_fraction": supported_fraction}, {}

    supported_weights = flat_loss_mask[supported]
    margins = (teacher_logits[supported] - competitor_logits[supported]).to(dtype=value_dtype)
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss, metrics = _margin_metrics(
        margins=margins,
        weights=supported_weights,
        violations=violations,
        requested_margin=float(margin),
        metric_prefix="teacher_same_family_action_margin",
    )
    metrics["teacher_same_family_action_margin_supported_fraction"] = supported_fraction
    return (
        loss.to(dtype=value_dtype),
        metrics,
        {"teacher_same_family_action_margins": margins.detach()},
    )


def dense_teacher_same_family_action_margin_loss(
    *,
    logits: Tensor,
    legal_mask: Tensor,
    action_family_ids: Tensor,
    teacher_action: Tensor,
    teacher_family: Tensor,
    teacher_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    zero: Tensor,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Dense-mask equivalent of the same-family teacher-action margin loss."""

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_action = teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_legal_mask = legal_mask.reshape(-1, legal_mask.shape[-1]).to(dtype=torch.bool)
    action_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
    action_row_indices = torch.nonzero(action_rows, as_tuple=False).squeeze(1)
    row_weights = flat_loss_mask[action_rows]
    if action_row_indices.numel() == 0 or float(row_weights.sum().item()) <= 0.0:
        return zero, {}, {}

    action_targets = flat_teacher_action[action_rows]
    action_families = flat_teacher_family[action_rows]
    in_range = (action_targets >= 0) & (action_targets < flat_logits.shape[-1])
    if not bool(in_range.any().item()):
        return zero, {"teacher_same_family_action_margin_supported_fraction": 0.0}, {}

    row_indices = action_row_indices[in_range]
    targets = action_targets[in_range]
    target_families = action_families[in_range]
    selected_logits = flat_logits[row_indices]
    selected_masks = flat_legal_mask[row_indices]
    selected_weights = flat_loss_mask[row_indices]
    action_family_ids = action_family_ids.to(device=flat_logits.device, dtype=torch.long)
    selected_same_family = selected_masks & (action_family_ids.unsqueeze(0) == target_families.unsqueeze(1))
    teacher_legal = selected_same_family.gather(1, targets.unsqueeze(1)).squeeze(1)
    competitor_logits = torch.where(selected_same_family, selected_logits, torch.full_like(selected_logits, -torch.inf))
    competitor_logits = competitor_logits.scatter(1, targets.unsqueeze(1), -torch.inf)
    competitor_max = competitor_logits.max(dim=1).values
    supported = teacher_legal & torch.isfinite(competitor_max)
    supported_fraction = float(selected_weights[supported].sum().item() / max(float(row_weights.sum().item()), 1.0e-8))
    if not bool(supported.any().item()):
        return zero, {"teacher_same_family_action_margin_supported_fraction": supported_fraction}, {}

    target_logits = selected_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    margins = (target_logits[supported] - competitor_max[supported]).to(dtype=logits.dtype)
    supported_weights = selected_weights[supported]
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss, metrics = _margin_metrics(
        margins=margins,
        weights=supported_weights,
        violations=violations,
        requested_margin=float(margin),
        metric_prefix="teacher_same_family_action_margin",
    )
    metrics["teacher_same_family_action_margin_supported_fraction"] = supported_fraction
    return (
        loss.to(dtype=logits.dtype),
        metrics,
        {"teacher_same_family_action_margins": margins.detach()},
    )


def packed_public_nonpass_over_pass_loss(
    *,
    packed_view: PackedStructuredLegalView,
    target_logits: Tensor,
    pass_action_id: int,
    teacher_valid: Tensor,
    loss_mask: Tensor,
    margin: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Hinge loss that follows public guidance when it ranks non-pass above pass."""

    if target_logits.shape != packed_view.logits.shape:
        raise ValueError("target logits must align with packed student logits")
    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    pass_candidates = packed_view.action_ids.to(dtype=torch.long) == int(pass_action_id)
    nonpass_candidates = ~pass_candidates
    neg_inf = torch.full_like(packed_view.logits, -torch.inf)

    student_pass_logits = segment_max(
        torch.where(pass_candidates, packed_view.logits, neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    student_nonpass_logits = segment_max(
        torch.where(nonpass_candidates, packed_view.logits, neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    target_pass_logits = segment_max(
        torch.where(pass_candidates, target_logits.to(dtype=packed_view.logits.dtype), neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    target_nonpass_logits = segment_max(
        torch.where(nonpass_candidates, target_logits.to(dtype=packed_view.logits.dtype), neg_inf),
        packed_view.row_indices,
        packed_view.row_count,
    )
    target_prefers_nonpass = target_nonpass_logits > target_pass_logits
    candidate_rows = (
        packed_view.row_has_candidates
        & flat_teacher_valid
        & torch.isfinite(student_pass_logits)
        & torch.isfinite(student_nonpass_logits)
        & torch.isfinite(target_pass_logits)
        & torch.isfinite(target_nonpass_logits)
    )
    row_weights = flat_loss_mask[candidate_rows]
    if not bool(candidate_rows.any().item()) or float(row_weights.sum().item()) <= 0.0:
        return zero, {}, {}
    supported = candidate_rows & target_prefers_nonpass
    supported_fraction = float(flat_loss_mask[supported].sum().item() / max(float(row_weights.sum().item()), 1.0e-8))
    if not bool(supported.any().item()):
        return zero, {"teacher_public_nonpass_over_pass_supported_fraction": supported_fraction}, {}

    supported_weights = flat_loss_mask[supported]
    margins = (student_nonpass_logits[supported] - student_pass_logits[supported]).to(dtype=value_dtype)
    violations = torch.relu(margins.new_tensor(float(margin)) - margins)
    loss, base_metrics = _margin_metrics(
        margins=margins,
        weights=supported_weights,
        violations=violations,
        requested_margin=float(margin),
        metric_prefix="teacher_action_margin",
    )
    metrics = {
        "teacher_public_nonpass_over_pass_loss": float(loss.detach().item()),
        "teacher_public_nonpass_over_pass_supported_fraction": supported_fraction,
        "teacher_public_nonpass_over_pass_margin_mean": base_metrics["teacher_action_margin_mean"],
        "teacher_public_nonpass_over_pass_satisfied_fraction": base_metrics["teacher_action_margin_satisfied_fraction"],
    }
    return loss.to(dtype=value_dtype), metrics, {"teacher_public_nonpass_over_pass_margins": margins.detach()}


__all__ = [
    "dense_teacher_action_margin_loss",
    "dense_teacher_same_family_action_margin_loss",
    "packed_public_nonpass_over_pass_loss",
    "packed_teacher_same_family_action_margin_loss",
    "packed_teacher_action_margin_loss",
]
