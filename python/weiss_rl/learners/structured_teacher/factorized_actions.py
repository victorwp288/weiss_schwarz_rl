"""Factorized exact-action and same-family teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True, slots=True)
class FactorizedTeacherActionSupervisionResult:
    action_loss: Tensor
    same_family_action_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_factorized_teacher_action_supervision(
    *,
    family_log_probs: Tensor,
    factorized_top_action_ids: Tensor | None,
    factorized_same_family_action_logp: Tensor | None,
    factorized_same_family_top_action_ids: Tensor | None,
    flat_teacher_action: Tensor | None,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    flat_loss_mask: Tensor,
    exact_action_family_rows: Tensor | None,
    play_family_id: int,
    move_family_id: int,
    action_coef: float,
    same_family_action_coef: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> FactorizedTeacherActionSupervisionResult:
    metrics: dict[str, float] = {}
    context: dict[str, Tensor] = {}
    action_loss = zero
    same_family_action_loss = zero

    if flat_teacher_action is not None and float(action_coef) != 0.0 and factorized_same_family_action_logp is not None:
        action_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if exact_action_family_rows is not None:
            action_rows = action_rows & exact_action_family_rows
        if bool(action_rows.any().item()):
            teacher_family_log_probs = family_log_probs.gather(
                1,
                torch.clamp(flat_teacher_family, min=0).unsqueeze(1),
            ).squeeze(1)
            teacher_action_log_probs = teacher_family_log_probs + factorized_same_family_action_logp.reshape(-1).to(
                dtype=value_dtype
            )
            row_weight = flat_loss_mask[action_rows]
            supported = action_rows & torch.isfinite(teacher_action_log_probs)
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_log_probs = teacher_action_log_probs[supported]
                supported_weights = flat_loss_mask[supported]
                action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                if factorized_top_action_ids is not None:
                    supported_predictions = factorized_top_action_ids.reshape(-1).to(dtype=torch.long)[supported]
                    supported_targets = flat_teacher_action[supported]
                    metrics["teacher_action_accuracy"] = float(
                        ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                metrics["teacher_action_loss"] = float(action_loss.detach().item())
                context["teacher_action_log_probs"] = supported_log_probs.detach()

    if (
        flat_teacher_action is not None
        and float(same_family_action_coef) != 0.0
        and factorized_same_family_action_logp is not None
        and factorized_same_family_top_action_ids is not None
    ):
        same_family_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if exact_action_family_rows is not None:
            same_family_rows = same_family_rows & exact_action_family_rows
        if bool(same_family_rows.any().item()):
            same_family_log_probs = factorized_same_family_action_logp.reshape(-1).to(dtype=value_dtype)
            same_family_top_actions = factorized_same_family_top_action_ids.reshape(-1).to(dtype=torch.long)
            row_weight = flat_loss_mask[same_family_rows]
            supported = same_family_rows & torch.isfinite(same_family_log_probs)
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_same_family_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_log_probs = same_family_log_probs[supported]
                supported_weights = flat_loss_mask[supported]
                supported_predictions = same_family_top_actions[supported]
                supported_targets = flat_teacher_action[supported]
                same_family_action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                metrics["teacher_same_family_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                context["teacher_same_family_action_log_probs"] = supported_log_probs.detach()
                supported_families = flat_teacher_family[supported]
                main_play_supported = supported_families == play_family_id
                if bool(main_play_supported.any().item()):
                    play_weights = supported_weights[main_play_supported]
                    metrics["teacher_same_family_main_play_character_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_play_supported] == supported_targets[main_play_supported]
                            ).float()
                            * play_weights
                        )
                        .sum()
                        .item()
                        / max(float(play_weights.sum().item()), 1.0)
                    )
                main_move_supported = supported_families == move_family_id
                if bool(main_move_supported.any().item()):
                    move_weights = supported_weights[main_move_supported]
                    metrics["teacher_same_family_main_move_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_move_supported] == supported_targets[main_move_supported]
                            ).float()
                            * move_weights
                        )
                        .sum()
                        .item()
                        / max(float(move_weights.sum().item()), 1.0)
                    )

    return FactorizedTeacherActionSupervisionResult(
        action_loss=action_loss,
        same_family_action_loss=same_family_action_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["FactorizedTeacherActionSupervisionResult", "compute_factorized_teacher_action_supervision"]
