"""Structured teacher branch dispatch preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView, packed_structured_legal_view
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics


@dataclass(frozen=True, slots=True)
class StructuredTeacherZeroContext:
    zero: Tensor
    value_dtype: torch.dtype
    empty_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class StructuredTeacherRequiredLabels:
    family: Tensor
    slot: Tensor
    attack_type: Tensor
    valid: Tensor


@dataclass(frozen=True, slots=True)
class StructuredTeacherBranch:
    use_factorized: bool
    use_packed: bool
    use_dense: bool


@dataclass(frozen=True, slots=True)
class StructuredTeacherDispatch:
    zero_context: StructuredTeacherZeroContext
    labels: StructuredTeacherRequiredLabels | None
    packed_view: PackedStructuredLegalView | None
    branch: StructuredTeacherBranch


def resolve_structured_teacher_zero_context(
    *,
    logits: Tensor | None,
    packed_view: PackedStructuredLegalView | None,
    loss_mask: Tensor,
) -> StructuredTeacherZeroContext:
    zero_source = logits
    if zero_source is None and packed_view is not None:
        zero_source = packed_view.logits
    if zero_source is None:
        zero_source = loss_mask
    zero = zero_source.sum() * 0.0
    return StructuredTeacherZeroContext(
        zero=zero,
        value_dtype=zero.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )


def resolve_structured_teacher_required_labels(
    *,
    teacher_family: Tensor | None,
    teacher_slot: Tensor | None,
    teacher_attack_type: Tensor | None,
    teacher_valid: Tensor | None,
) -> StructuredTeacherRequiredLabels | None:
    if teacher_family is None or teacher_slot is None or teacher_attack_type is None or teacher_valid is None:
        return None
    return StructuredTeacherRequiredLabels(
        family=teacher_family,
        slot=teacher_slot,
        attack_type=teacher_attack_type,
        valid=teacher_valid,
    )


def resolve_structured_teacher_branch(
    *,
    factorized_family_log_probs: Tensor | None,
    packed_view: PackedStructuredLegalView | None,
    logits: Tensor | None,
    legal_mask: Tensor | None,
) -> StructuredTeacherBranch:
    use_factorized = factorized_family_log_probs is not None
    use_packed = (not use_factorized) and packed_view is not None
    use_dense = (not use_factorized) and (not use_packed) and logits is not None and legal_mask is not None
    return StructuredTeacherBranch(
        use_factorized=use_factorized,
        use_packed=use_packed,
        use_dense=use_dense,
    )


def resolve_structured_teacher_dispatch(
    *,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    packed_ids: Tensor | None,
    packed_offsets: Tensor | None,
    packed_meta: Tensor | None,
    packed_view: PackedStructuredLegalView | None,
    factorized_family_log_probs: Tensor | None,
    teacher_family: Tensor | None,
    teacher_slot: Tensor | None,
    teacher_attack_type: Tensor | None,
    teacher_valid: Tensor | None,
    loss_mask: Tensor,
) -> StructuredTeacherDispatch:
    zero_context = resolve_structured_teacher_zero_context(
        logits=logits,
        packed_view=packed_view,
        loss_mask=loss_mask,
    )
    labels = resolve_structured_teacher_required_labels(
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_valid=teacher_valid,
    )
    if labels is None:
        return StructuredTeacherDispatch(
            zero_context=zero_context,
            labels=None,
            packed_view=packed_view,
            branch=StructuredTeacherBranch(
                use_factorized=False,
                use_packed=False,
                use_dense=False,
            ),
        )

    resolved_packed_view = (
        packed_view
        if packed_view is not None
        else packed_structured_legal_view(
            logits=logits,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    )
    branch = resolve_structured_teacher_branch(
        factorized_family_log_probs=factorized_family_log_probs,
        packed_view=resolved_packed_view,
        logits=logits,
        legal_mask=legal_mask,
    )
    return StructuredTeacherDispatch(
        zero_context=zero_context,
        labels=labels,
        packed_view=resolved_packed_view,
        branch=branch,
    )


__all__ = [
    "StructuredTeacherBranch",
    "StructuredTeacherDispatch",
    "StructuredTeacherRequiredLabels",
    "StructuredTeacherZeroContext",
    "resolve_structured_teacher_branch",
    "resolve_structured_teacher_dispatch",
    "resolve_structured_teacher_required_labels",
    "resolve_structured_teacher_zero_context",
]
