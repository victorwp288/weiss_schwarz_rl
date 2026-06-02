"""Factorized hand/arg0 teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True, slots=True)
class FactorizedTeacherHandSupervisionResult:
    hand_loss: Tensor
    metrics: dict[str, float]


def compute_factorized_teacher_hand_supervision(
    *,
    factorized_same_family_arg0_logp: Tensor | None,
    factorized_same_family_top_arg0: Tensor | None,
    flat_teacher_action: Tensor | None,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    flat_loss_mask: Tensor,
    exact_action_family_rows: Tensor | None,
    hand_targets_by_action: Tensor,
    hand_family_ids: tuple[int, ...],
    play_family_id: int,
    clock_from_hand_family_id: int,
    hand_coef: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> FactorizedTeacherHandSupervisionResult:
    metrics: dict[str, float] = {}
    hand_loss = zero

    if (
        flat_teacher_action is None
        or float(hand_coef) == 0.0
        or factorized_same_family_arg0_logp is None
        or factorized_same_family_top_arg0 is None
        or not hand_family_ids
    ):
        return FactorizedTeacherHandSupervisionResult(hand_loss=hand_loss, metrics=metrics)

    valid_action_rows = (flat_teacher_action >= 0) & (flat_teacher_action < int(hand_targets_by_action.shape[0]))
    hand_targets = torch.full_like(flat_teacher_action, -1)
    if bool(valid_action_rows.any().item()):
        hand_targets[valid_action_rows] = hand_targets_by_action.index_select(
            0,
            flat_teacher_action[valid_action_rows],
        )

    hand_rows = flat_teacher_valid & (hand_targets >= 0)
    hand_rows = hand_rows & torch.isin(
        flat_teacher_family,
        torch.as_tensor(hand_family_ids, device=flat_teacher_family.device, dtype=flat_teacher_family.dtype),
    )
    if exact_action_family_rows is not None:
        hand_rows = hand_rows & exact_action_family_rows

    same_family_arg0_logp = factorized_same_family_arg0_logp.reshape(-1).to(dtype=value_dtype)
    same_family_top_arg0 = factorized_same_family_top_arg0.reshape(-1).to(dtype=torch.long)
    supported = hand_rows & torch.isfinite(same_family_arg0_logp)
    if bool(hand_rows.any().item()):
        row_weight = flat_loss_mask[hand_rows]
        if float(row_weight.sum().item()) > 0.0:
            metrics["teacher_hand_supported_fraction"] = float(
                (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
            )
    if bool(supported.any().item()):
        supported_weights = flat_loss_mask[supported]
        supported_targets = hand_targets[supported]
        supported_predictions = same_family_top_arg0[supported]
        hand_loss = weighted_mean(-same_family_arg0_logp[supported], supported_weights).to(dtype=value_dtype)
        metrics["teacher_hand_loss"] = float(hand_loss.detach().item())
        metrics["teacher_hand_accuracy"] = float(
            ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
            / max(float(supported_weights.sum().item()), 1.0)
        )
        supported_families = flat_teacher_family[supported]
        main_play_supported = supported_families == play_family_id
        if bool(main_play_supported.any().item()):
            play_weights = supported_weights[main_play_supported]
            metrics["teacher_main_play_character_hand_accuracy"] = float(
                (
                    (supported_predictions[main_play_supported] == supported_targets[main_play_supported]).float()
                    * play_weights
                )
                .sum()
                .item()
                / max(float(play_weights.sum().item()), 1.0)
            )
        clock_supported = supported_families == clock_from_hand_family_id
        if bool(clock_supported.any().item()):
            clock_weights = supported_weights[clock_supported]
            metrics["teacher_clock_from_hand_accuracy"] = float(
                ((supported_predictions[clock_supported] == supported_targets[clock_supported]).float() * clock_weights)
                .sum()
                .item()
                / max(float(clock_weights.sum().item()), 1.0)
            )

    return FactorizedTeacherHandSupervisionResult(hand_loss=hand_loss, metrics=metrics)


__all__ = ["FactorizedTeacherHandSupervisionResult", "compute_factorized_teacher_hand_supervision"]
