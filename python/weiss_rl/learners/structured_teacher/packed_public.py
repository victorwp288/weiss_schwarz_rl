"""Packed public-heuristic teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView, packed_soft_target_cross_entropy
from weiss_rl.learners.structured_teacher.margin import packed_public_nonpass_over_pass_loss
from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True, slots=True)
class PackedTeacherPublicSupervisionResult:
    public_heuristic_loss: Tensor
    public_nonpass_over_pass_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_packed_teacher_public_supervision(
    *,
    packed_view: PackedStructuredLegalView,
    public_heuristic_target_logits: Tensor | None,
    public_heuristic_family_ids: tuple[int, ...],
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    flat_loss_mask: Tensor,
    pass_action_id: int,
    public_heuristic_coef: float,
    public_heuristic_temperature: float,
    public_nonpass_over_pass_coef: float,
    public_nonpass_over_pass_margin: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> PackedTeacherPublicSupervisionResult:
    metrics: dict[str, float] = {}
    context: dict[str, Tensor] = {}
    public_heuristic_loss = zero
    public_nonpass_over_pass_loss = zero

    if public_heuristic_target_logits is not None and float(public_heuristic_coef) != 0.0:
        public_rows = packed_view.row_has_candidates & flat_teacher_valid
        if public_heuristic_family_ids:
            public_rows = public_rows & torch.isin(
                flat_teacher_family,
                torch.as_tensor(
                    public_heuristic_family_ids,
                    device=flat_teacher_family.device,
                    dtype=flat_teacher_family.dtype,
                ),
            )
        if bool(public_rows.any().item()):
            row_cross_entropy, row_student_top_mass, row_target_entropy = packed_soft_target_cross_entropy(
                packed_view,
                target_logits=public_heuristic_target_logits,
                temperature=float(public_heuristic_temperature),
            )
            public_weights = flat_loss_mask[public_rows]
            if float(public_weights.sum().item()) > 0.0:
                metrics["teacher_public_heuristic_supported_fraction"] = 1.0
                metrics["teacher_public_heuristic_top1_mass"] = float(
                    weighted_mean(row_student_top_mass[public_rows], public_weights).item()
                )
                metrics["teacher_public_heuristic_target_entropy"] = float(
                    weighted_mean(row_target_entropy[public_rows], public_weights).item()
                )
                public_heuristic_loss = weighted_mean(
                    row_cross_entropy[public_rows],
                    public_weights,
                ).to(dtype=value_dtype)
                metrics["teacher_public_heuristic_loss"] = float(public_heuristic_loss.detach().item())

    if public_heuristic_target_logits is not None and float(public_nonpass_over_pass_coef) != 0.0:
        (
            public_nonpass_over_pass_loss,
            public_nonpass_over_pass_metrics,
            public_nonpass_over_pass_context,
        ) = packed_public_nonpass_over_pass_loss(
            packed_view=packed_view,
            target_logits=public_heuristic_target_logits,
            pass_action_id=int(pass_action_id),
            teacher_valid=flat_teacher_valid,
            loss_mask=flat_loss_mask,
            margin=float(public_nonpass_over_pass_margin),
            zero=zero,
            value_dtype=value_dtype,
        )
        metrics.update(public_nonpass_over_pass_metrics)
        context.update(public_nonpass_over_pass_context)

    return PackedTeacherPublicSupervisionResult(
        public_heuristic_loss=public_heuristic_loss,
        public_nonpass_over_pass_loss=public_nonpass_over_pass_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["PackedTeacherPublicSupervisionResult", "compute_packed_teacher_public_supervision"]
