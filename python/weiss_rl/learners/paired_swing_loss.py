"""Paired-swing contrastive losses for replay repair batches."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.learners.action_logp import packed_selected_action_logp
from weiss_rl.learners.tensor_ops import weighted_mean


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

    if positive_actions.shape != negative_actions.shape:
        raise ValueError("positive_actions and negative_actions must have the same shape")
    if positive_actions.shape != loss_mask.shape:
        raise ValueError("paired-swing action tensors must match loss_mask shape")
    if negative_valid.shape != loss_mask.shape:
        raise ValueError("negative_valid must match loss_mask shape")

    normalized_scope = str(loss_scope).strip().lower()
    if normalized_scope not in {"row", "episode_mean", "label_mean"}:
        raise ValueError("paired_swing loss_scope must be one of: row, episode_mean, label_mean")
    normalized_compare_to = str(compare_to).strip().lower()
    if normalized_compare_to not in {"negative", "top_other"}:
        raise ValueError("paired_swing compare_to must be one of: negative, top_other")
    if reference_packed_logits is not None and reference_packed_logits.shape != packed_logits.shape:
        raise ValueError("reference_packed_logits must match packed_logits shape")
    if float(margin_retention_coef) < 0.0:
        raise ValueError("paired_swing margin_retention_coef must be >= 0")
    if float(margin_retention_margin) < 0.0:
        raise ValueError("paired_swing margin_retention_margin must be >= 0")
    if float(top_action_retention_coef) < 0.0:
        raise ValueError("paired_swing top_action_retention_coef must be >= 0")
    if float(top_action_retention_margin) < 0.0:
        raise ValueError("paired_swing top_action_retention_margin must be >= 0")

    zero = packed_logits.sum() * 0.0
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
    metrics: dict[str, float] = {
        f"{metric_prefix}_candidate_rows": float(active_rows.sum().item()),
        f"{metric_prefix}_distinct_fraction": (
            0.0 if train_weight_total <= 0.0 else raw_weight_total / max(train_weight_total, 1.0e-8)
        ),
    }
    if not bool(active_rows.any().item()) or raw_weight_total <= 0.0:
        metrics.update(
            {
                f"{metric_prefix}_rows": 0.0,
                f"{metric_prefix}_supported_fraction": 0.0,
                f"{metric_prefix}_loss": 0.0,
            }
        )
        return zero, metrics, {}

    if normalized_compare_to == "top_other":
        margin_by_row, supported, positive_logp_by_row, negative_logp_by_row = _positive_vs_top_other_margin_by_row(
            packed_logits=packed_logits,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            flat_positive_actions=flat_positive_actions,
            active_rows=active_rows,
        )
    else:
        positive_logp = packed_selected_action_logp(
            packed_logits,
            legal_ids,
            legal_offsets,
            flat_positive_actions.reshape_as(positive_actions),
            pass_action_id=pass_action_id,
            strict=False,
        ).reshape(-1)
        negative_logp = packed_selected_action_logp(
            packed_logits,
            legal_ids,
            legal_offsets,
            flat_negative_actions.reshape_as(negative_actions),
            pass_action_id=pass_action_id,
            strict=False,
        ).reshape(-1)
        supported = active_rows & torch.isfinite(positive_logp) & torch.isfinite(negative_logp)
        margin_by_row = (positive_logp - negative_logp).to(dtype=packed_logits.dtype)
        positive_logp_by_row = positive_logp
        negative_logp_by_row = negative_logp
    margins = margin_by_row[supported].to(dtype=packed_logits.dtype)
    positive_metric_logp = positive_logp_by_row[supported]
    negative_metric_logp = negative_logp_by_row[supported]
    supported_weight = flat_loss_mask[supported]
    supported_weight_total = float(supported_weight.sum().item()) if bool(supported.any().item()) else 0.0
    metrics[f"{metric_prefix}_supported_fraction"] = supported_weight_total / max(raw_weight_total, 1.0e-8)
    metrics[f"{metric_prefix}_rows"] = float(supported.sum().item())
    if not bool(supported.any().item()) or supported_weight_total <= 0.0:
        metrics[f"{metric_prefix}_loss"] = 0.0
        return zero, metrics, {}

    if normalized_scope == "episode_mean":
        loss, margin_mean, satisfied_fraction, scope_metrics = _episode_mean_margin_loss(
            margins=margins,
            supported_weight=supported_weight,
            supported=supported,
            episode_count=int(positive_actions.shape[1]),
            margin=float(margin),
        )
    elif normalized_scope == "label_mean":
        if group_ids is None:
            raise ValueError("paired_swing loss_scope label_mean requires group_ids")
        if group_ids.shape != loss_mask.shape:
            raise ValueError("paired-swing group_ids must match loss_mask shape")
        loss, margin_mean, satisfied_fraction, scope_metrics = _group_mean_margin_loss(
            margins=margins,
            group_ids=group_ids.reshape(-1).to(device=packed_logits.device, dtype=torch.long)[supported],
            margin=float(margin),
        )
    else:
        violations = torch.relu(margins.new_tensor(float(margin)) - margins)
        loss = weighted_mean(violations, supported_weight).to(dtype=packed_logits.dtype)
        margin_mean = weighted_mean(margins, supported_weight)
        satisfied_fraction = (
            (margins >= float(margin)).to(dtype=supported_weight.dtype) * supported_weight
        ).sum() / max(supported_weight_total, 1.0e-8)
        scope_metrics = {}
    retention_loss, retention_metrics, retention_tensors = _margin_retention_loss_and_metrics(
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
    top_retention_loss, top_retention_metrics, top_retention_tensors = _top_action_retention_loss_and_metrics(
        packed_logits=packed_logits,
        reference_packed_logits=reference_packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        supported=supported,
        supported_weight=flat_loss_mask,
        retention_margin=float(top_action_retention_margin),
        metric_prefix=metric_prefix,
    )
    if float(top_action_retention_coef) != 0.0:
        loss = loss + (top_retention_loss * float(top_action_retention_coef))
    metrics.update(
        {
            f"{metric_prefix}_loss": float(loss.detach().item()),
            f"{metric_prefix}_margin_mean": float(margin_mean.detach().item()),
            f"{metric_prefix}_satisfied_fraction": float(satisfied_fraction.detach().item()),
            f"{metric_prefix}_loss_scope_episode_mean": 1.0 if normalized_scope == "episode_mean" else 0.0,
            f"{metric_prefix}_loss_scope_label_mean": 1.0 if normalized_scope == "label_mean" else 0.0,
            f"{metric_prefix}_compare_to_top_other": 1.0 if normalized_compare_to == "top_other" else 0.0,
            f"{metric_prefix}_margin_retention_coef": float(margin_retention_coef),
            f"{metric_prefix}_margin_retention_margin": float(margin_retention_margin),
            f"{metric_prefix}_top_action_retention_coef": float(top_action_retention_coef),
            f"{metric_prefix}_top_action_retention_margin": float(top_action_retention_margin),
            f"{metric_prefix}_positive_logp_mean": float(
                weighted_mean(positive_metric_logp, supported_weight).detach().item()
            ),
            f"{metric_prefix}_negative_logp_mean": float(
                weighted_mean(negative_metric_logp, supported_weight).detach().item()
            ),
            **scope_metrics,
        }
    )
    metrics.update(retention_metrics)
    metrics.update(top_retention_metrics)
    tensors = {"paired_swing_margins": margins.detach()}
    tensors.update(retention_tensors)
    tensors.update(top_retention_tensors)
    return loss, metrics, tensors


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
    return _top_action_retention_loss_and_metrics(
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
    margin_by_row, supported, target_logp_by_row, best_other_logp_by_row = _positive_vs_top_other_margin_by_row(
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


def _positive_vs_top_other_margin_by_row(
    *,
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    flat_positive_actions: Tensor,
    active_rows: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_indices = torch.nonzero(active_rows, as_tuple=False).reshape(-1)
    supported = torch.zeros_like(active_rows, dtype=torch.bool)
    margin_by_row = torch.full_like(active_rows.to(dtype=packed_logits.dtype), -torch.inf)
    positive_logp_by_row = torch.full_like(margin_by_row, -torch.inf)
    top_other_logp_by_row = torch.full_like(margin_by_row, -torch.inf)
    offsets = legal_offsets.to(device=packed_logits.device, dtype=torch.long)
    ids = legal_ids.to(device=packed_logits.device, dtype=torch.long)
    for row_index_tensor in row_indices:
        row_index = int(row_index_tensor.detach().cpu().item())
        start = int(offsets[row_index].detach().cpu().item())
        stop = int(offsets[row_index + 1].detach().cpu().item())
        if stop <= start + 1:
            continue
        row_ids = ids[start:stop]
        positive_action = flat_positive_actions[row_index]
        positive_matches = row_ids == positive_action
        if not bool(positive_matches.any().item()):
            continue
        row_logp = torch.log_softmax(packed_logits[start:stop], dim=0)
        positive_logp = row_logp[positive_matches].max()
        top_other_logp = row_logp.masked_fill(positive_matches, float("-inf")).max()
        if not bool(torch.isfinite(positive_logp).item()) or not bool(torch.isfinite(top_other_logp).item()):
            continue
        supported[row_index] = True
        positive_logp_by_row[row_index] = positive_logp
        top_other_logp_by_row[row_index] = top_other_logp
        margin_by_row[row_index] = positive_logp - top_other_logp
    return margin_by_row, supported, positive_logp_by_row, top_other_logp_by_row


def _margin_retention_loss_and_metrics(
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
        return zero, _empty_retention_metrics(metric_prefix=metric_prefix), {}
    if compare_to == "top_other":
        reference_margin_by_row, reference_supported, _positive_logp, _negative_logp = (
            _positive_vs_top_other_margin_by_row(
                packed_logits=reference_packed_logits,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                flat_positive_actions=flat_positive_actions,
                active_rows=active_rows,
            )
        )
    else:
        reference_positive_logp = packed_selected_action_logp(
            reference_packed_logits,
            legal_ids,
            legal_offsets,
            flat_positive_actions.reshape_as(positive_actions),
            pass_action_id=pass_action_id,
            strict=False,
        ).reshape(-1)
        reference_negative_logp = packed_selected_action_logp(
            reference_packed_logits,
            legal_ids,
            legal_offsets,
            flat_negative_actions.reshape_as(negative_actions),
            pass_action_id=pass_action_id,
            strict=False,
        ).reshape(-1)
        reference_margin_by_row = (reference_positive_logp - reference_negative_logp).to(
            dtype=current_margin_by_row.dtype
        )
        reference_supported = (
            active_rows & torch.isfinite(reference_positive_logp) & torch.isfinite(reference_negative_logp)
        )
    retention_supported = supported & reference_supported & torch.isfinite(current_margin_by_row)
    row_count = int(retention_supported.sum().detach().cpu().item())
    if row_count <= 0:
        return zero, _empty_retention_metrics(metric_prefix=metric_prefix), {}
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


def _empty_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_margin_retention_loss": 0.0,
        f"{metric_prefix}_margin_retention_rows": 0.0,
        f"{metric_prefix}_margin_retention_violation_fraction": 0.0,
        f"{metric_prefix}_margin_delta_mean": 0.0,
        f"{metric_prefix}_margin_delta_min": 0.0,
    }


def _top_action_retention_loss_and_metrics(
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
    zero = packed_logits.sum() * 0.0
    if reference_packed_logits is None:
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
    row_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    if int(row_indices.numel()) <= 0:
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
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
        return zero, _empty_top_action_retention_metrics(metric_prefix=metric_prefix), {}
    gap_tensor = torch.stack(gaps).to(dtype=packed_logits.dtype)
    weight_tensor = torch.stack(weights).to(device=packed_logits.device, dtype=torch.float32)
    agreement_tensor = torch.stack(agreements).to(device=packed_logits.device, dtype=torch.float32)
    violations = torch.relu(gap_tensor.new_tensor(float(retention_margin)) - gap_tensor)
    loss = weighted_mean(violations, weight_tensor).to(dtype=packed_logits.dtype)
    violation_fraction = weighted_mean((violations > 0.0).to(dtype=weight_tensor.dtype), weight_tensor)
    agreement_fraction = weighted_mean(agreement_tensor, weight_tensor)
    metrics = {
        f"{metric_prefix}_top_action_retention_loss": float(loss.detach().item()),
        f"{metric_prefix}_top_action_retention_rows": float(gap_tensor.numel()),
        f"{metric_prefix}_top_action_retention_violation_fraction": float(violation_fraction.detach().item()),
        f"{metric_prefix}_top_action_retention_gap_mean": float(
            weighted_mean(gap_tensor, weight_tensor).detach().item()
        ),
        f"{metric_prefix}_top_action_retention_gap_min": float(gap_tensor.detach().min().item()),
        f"{metric_prefix}_top_action_retention_agreement_fraction": float(agreement_fraction.detach().item()),
    }
    return loss, metrics, {f"{metric_prefix}_top_action_retention_gap": gap_tensor.detach()}


def _empty_top_action_retention_metrics(*, metric_prefix: str) -> dict[str, float]:
    return {
        f"{metric_prefix}_top_action_retention_loss": 0.0,
        f"{metric_prefix}_top_action_retention_rows": 0.0,
        f"{metric_prefix}_top_action_retention_violation_fraction": 0.0,
        f"{metric_prefix}_top_action_retention_gap_mean": 0.0,
        f"{metric_prefix}_top_action_retention_gap_min": 0.0,
        f"{metric_prefix}_top_action_retention_agreement_fraction": 0.0,
    }


def _group_mean_margin_loss(
    *,
    margins: Tensor,
    group_ids: Tensor,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    if group_ids.shape != margins.shape:
        raise ValueError("paired-swing group_ids must match supported margins")
    unique_group_ids = torch.unique(group_ids, sorted=True)
    group_margins: list[Tensor] = []
    for group_id in unique_group_ids:
        group_mask = group_ids == group_id
        group_margins.append(margins[group_mask].mean())
    if not group_margins:
        zero = margins.sum() * 0.0
        return zero, zero, zero, {"paired_swing_label_count": 0.0, "paired_swing_label_rows_mean": 0.0}
    stacked_margins = torch.stack(group_margins).to(dtype=margins.dtype)
    violations = torch.relu(stacked_margins.new_tensor(float(margin)) - stacked_margins)
    loss = violations.mean().to(dtype=margins.dtype)
    margin_mean = stacked_margins.mean()
    satisfied_fraction = (stacked_margins >= float(margin)).to(dtype=margins.dtype).mean()
    return (
        loss,
        margin_mean,
        satisfied_fraction,
        {
            "paired_swing_label_count": float(unique_group_ids.numel()),
            "paired_swing_label_rows_mean": float(margins.numel() / max(int(unique_group_ids.numel()), 1)),
        },
    )


def _episode_mean_margin_loss(
    *,
    margins: Tensor,
    supported_weight: Tensor,
    supported: Tensor,
    episode_count: int,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    supported_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    episode_ids = torch.remainder(supported_indices, int(episode_count))
    unique_episode_ids = torch.unique(episode_ids, sorted=True)
    episode_margins: list[Tensor] = []
    episode_weights: list[Tensor] = []
    for episode_id in unique_episode_ids:
        episode_mask = episode_ids == episode_id
        weights = supported_weight[episode_mask]
        episode_margins.append(weighted_mean(margins[episode_mask], weights))
        episode_weights.append(weights.sum())
    stacked_margins = torch.stack(episode_margins).to(dtype=margins.dtype)
    stacked_weights = torch.stack(episode_weights).to(dtype=supported_weight.dtype)
    violations = torch.relu(stacked_margins.new_tensor(float(margin)) - stacked_margins)
    loss = weighted_mean(violations, stacked_weights).to(dtype=margins.dtype)
    margin_mean = weighted_mean(stacked_margins, stacked_weights)
    satisfied_fraction = (
        (stacked_margins >= float(margin)).to(dtype=stacked_weights.dtype) * stacked_weights
    ).sum() / torch.clamp(stacked_weights.sum(), min=1.0e-8)
    return (
        loss,
        margin_mean,
        satisfied_fraction,
        {
            "paired_swing_episode_count": float(unique_episode_ids.numel()),
            "paired_swing_episode_rows_mean": float(
                (supported_weight.sum() / max(int(unique_episode_ids.numel()), 1)).detach().item()
            ),
        },
    )


__all__ = [
    "packed_paired_swing_margin_loss",
    "packed_target_action_retention_loss",
    "packed_top_action_retention_loss",
]
