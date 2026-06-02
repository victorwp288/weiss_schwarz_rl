"""Packed teacher action-margin supervision orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.structured_teacher.margin import (
    packed_teacher_action_margin_loss,
    packed_teacher_same_family_action_margin_loss,
)


@dataclass(frozen=True, slots=True)
class PackedTeacherMarginSupervisionResult:
    action_margin_loss: Tensor
    same_family_action_margin_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_packed_teacher_margin_supervision(
    *,
    packed_view: PackedStructuredLegalView,
    flat_teacher_action: Tensor | None,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    flat_loss_mask: Tensor,
    exact_action_family_rows: Tensor | None,
    action_margin_coef: float,
    action_margin: float,
    same_family_action_margin_coef: float,
    same_family_action_margin: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> PackedTeacherMarginSupervisionResult:
    metrics: dict[str, float] = {}
    context: dict[str, Tensor] = {}
    teacher_valid = (
        flat_teacher_valid if exact_action_family_rows is None else (flat_teacher_valid & exact_action_family_rows)
    )

    action_margin_loss = zero
    if flat_teacher_action is not None and float(action_margin_coef) != 0.0:
        action_margin_loss, action_margin_metrics, action_margin_context = packed_teacher_action_margin_loss(
            packed_view=packed_view,
            teacher_action=flat_teacher_action,
            teacher_valid=teacher_valid,
            loss_mask=flat_loss_mask,
            margin=float(action_margin),
            zero=zero,
            value_dtype=value_dtype,
        )
        metrics.update(action_margin_metrics)
        context.update(action_margin_context)

    same_family_action_margin_loss = zero
    if flat_teacher_action is not None and float(same_family_action_margin_coef) != 0.0:
        same_family_action_margin_loss, same_family_margin_metrics, same_family_margin_context = (
            packed_teacher_same_family_action_margin_loss(
                packed_view=packed_view,
                teacher_action=flat_teacher_action,
                teacher_family=flat_teacher_family,
                teacher_valid=teacher_valid,
                loss_mask=flat_loss_mask,
                margin=float(same_family_action_margin),
                zero=zero,
                value_dtype=value_dtype,
            )
        )
        metrics.update(same_family_margin_metrics)
        context.update(same_family_margin_context)

    return PackedTeacherMarginSupervisionResult(
        action_margin_loss=action_margin_loss,
        same_family_action_margin_loss=same_family_action_margin_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["PackedTeacherMarginSupervisionResult", "compute_packed_teacher_margin_supervision"]
