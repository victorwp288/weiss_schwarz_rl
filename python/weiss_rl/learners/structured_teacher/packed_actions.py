"""Packed exact-action and same-family teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.action_logp import packed_selected_action_logp, packed_subset_action_logp_and_top_action
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.tensor_ops import segment_max, weighted_mean


@dataclass(frozen=True, slots=True)
class PackedTeacherActionSupervisionResult:
    action_loss: Tensor
    same_family_action_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_packed_teacher_action_supervision(
    *,
    packed_view: PackedStructuredLegalView,
    packed_offsets: Tensor | None,
    flat_teacher_action: Tensor | None,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    flat_loss_mask: Tensor,
    exact_action_family_rows: Tensor | None,
    play_family_id: int,
    move_family_id: int,
    action_catalog: ActionCatalog,
    action_coef: float,
    same_family_action_coef: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> PackedTeacherActionSupervisionResult:
    metrics: dict[str, float] = {}
    context: dict[str, Tensor] = {}
    action_loss = zero
    same_family_action_loss = zero

    if flat_teacher_action is not None and float(action_coef) != 0.0:
        action_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_action >= 0)
        if exact_action_family_rows is not None:
            action_rows = action_rows & exact_action_family_rows
        if bool(action_rows.any().item()):
            teacher_action_log_probs = (
                packed_selected_action_logp(
                    packed_view.logits,
                    packed_view.action_ids,
                    packed_offsets
                    if packed_offsets is not None
                    else packed_view.row_indices.new_zeros((packed_view.row_count + 1,)),
                    flat_teacher_action,
                    pass_action_id=int(action_catalog.pass_action_id),
                    strict=False,
                )
                .reshape(-1)
                .to(dtype=value_dtype)
            )
            supported = action_rows & torch.isfinite(teacher_action_log_probs)
            row_weight = flat_loss_mask[action_rows]
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_log_probs = teacher_action_log_probs[supported]
                supported_weights = flat_loss_mask[supported]
                action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                top_logits = segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
                top_matches = packed_view.logits >= (top_logits.index_select(0, packed_view.row_indices) - 1.0e-6)
                top_action_ids = torch.full(
                    (packed_view.row_count,),
                    -1,
                    dtype=torch.long,
                    device=packed_view.logits.device,
                )
                top_action_ids.scatter_reduce_(
                    0,
                    packed_view.row_indices.to(dtype=torch.long),
                    torch.where(
                        top_matches,
                        packed_view.action_ids.to(dtype=torch.long),
                        torch.full_like(packed_view.action_ids, -1),
                    ),
                    reduce="amax",
                    include_self=True,
                )
                metrics["teacher_action_accuracy"] = float(
                    ((top_action_ids[supported] == flat_teacher_action[supported]).float() * supported_weights)
                    .sum()
                    .item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_action_loss"] = float(action_loss.detach().item())
                context["teacher_action_log_probs"] = supported_log_probs.detach()

    if flat_teacher_action is not None and float(same_family_action_coef) != 0.0:
        same_family_rows = (
            packed_view.row_has_candidates
            & flat_teacher_valid
            & (flat_teacher_action >= 0)
            & (flat_teacher_family >= 0)
        )
        if exact_action_family_rows is not None:
            same_family_rows = same_family_rows & exact_action_family_rows
        if bool(same_family_rows.any().item()):
            candidate_mask = packed_view.family_ids == flat_teacher_family.index_select(
                0,
                packed_view.row_indices.to(dtype=torch.long),
            )
            same_family_log_probs, same_family_top_actions = packed_subset_action_logp_and_top_action(
                packed_view,
                flat_teacher_action,
                candidate_mask=candidate_mask,
                strict=False,
            )
            same_family_log_probs = same_family_log_probs.reshape(-1).to(dtype=value_dtype)
            same_family_top_actions = same_family_top_actions.reshape(-1).to(dtype=torch.long)
            supported = same_family_rows & torch.isfinite(same_family_log_probs)
            row_weight = flat_loss_mask[same_family_rows]
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_same_family_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_weights = flat_loss_mask[supported]
                supported_targets = flat_teacher_action[supported]
                same_family_action_loss = weighted_mean(
                    -same_family_log_probs[supported],
                    supported_weights,
                ).to(dtype=value_dtype)
                metrics["teacher_same_family_action_accuracy"] = float(
                    ((same_family_top_actions[supported] == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                context["teacher_same_family_action_log_probs"] = same_family_log_probs[supported].detach()
                main_play_supported = supported & (flat_teacher_family == play_family_id)
                if bool(main_play_supported.any().item()):
                    main_play_weights = flat_loss_mask[main_play_supported]
                    metrics["teacher_same_family_main_play_character_accuracy"] = float(
                        (
                            (
                                same_family_top_actions[main_play_supported] == flat_teacher_action[main_play_supported]
                            ).float()
                            * main_play_weights
                        )
                        .sum()
                        .item()
                        / max(float(main_play_weights.sum().item()), 1.0)
                    )
                main_move_supported = supported & (flat_teacher_family == move_family_id)
                if bool(main_move_supported.any().item()):
                    main_move_weights = flat_loss_mask[main_move_supported]
                    metrics["teacher_same_family_main_move_accuracy"] = float(
                        (
                            (
                                same_family_top_actions[main_move_supported] == flat_teacher_action[main_move_supported]
                            ).float()
                            * main_move_weights
                        )
                        .sum()
                        .item()
                        / max(float(main_move_weights.sum().item()), 1.0)
                    )

    return PackedTeacherActionSupervisionResult(
        action_loss=action_loss,
        same_family_action_loss=same_family_action_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["PackedTeacherActionSupervisionResult", "compute_packed_teacher_action_supervision"]
