"""Shared helpers for structured teacher-auxiliary loss implementations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import resolve_public_heuristic_family_ids


@dataclass(frozen=True, slots=True)
class FlatStructuredTeacherLabels:
    loss_mask: Tensor
    family: Tensor
    slot: Tensor
    move_source: Tensor | None
    attack_type: Tensor
    action: Tensor | None
    valid: Tensor


@dataclass(frozen=True, slots=True)
class StructuredTeacherAuxiliaryLossTerms:
    family: Tensor
    slot: Tensor
    attack_type: Tensor
    action: Tensor
    same_family_action: Tensor
    action_margin: Tensor
    same_family_action_margin: Tensor
    hand: Tensor | None = None
    move_source: Tensor | None = None
    public_heuristic: Tensor | None = None
    public_nonpass_over_pass: Tensor | None = None


@dataclass(frozen=True, slots=True)
class StructuredTeacherAuxiliaryCoefficients:
    family: float
    slot: float
    attack_type: float
    action: float
    same_family_action: float
    action_margin: float
    same_family_action_margin: float
    hand: float = 0.0
    move_source: float = 0.0
    public_heuristic: float = 0.0
    public_nonpass_over_pass: float = 0.0


def empty_structured_teacher_metrics() -> dict[str, float]:
    return {
        "teacher_active_fraction": 0.0,
        "teacher_valid_fraction": 0.0,
        "teacher_main_play_character_fraction": 0.0,
        "teacher_main_move_fraction": 0.0,
        "teacher_attack_fraction": 0.0,
        "teacher_family_accuracy": 0.0,
        "teacher_slot_accuracy": 0.0,
        "teacher_main_play_character_slot_accuracy": 0.0,
        "teacher_hand_accuracy": 0.0,
        "teacher_main_play_character_hand_accuracy": 0.0,
        "teacher_clock_from_hand_accuracy": 0.0,
        "teacher_move_source_accuracy": 0.0,
        "teacher_attack_type_accuracy": 0.0,
        "teacher_action_accuracy": 0.0,
        "teacher_same_family_action_accuracy": 0.0,
        "teacher_same_family_main_play_character_accuracy": 0.0,
        "teacher_same_family_main_move_accuracy": 0.0,
        "teacher_family_loss": 0.0,
        "teacher_slot_loss": 0.0,
        "teacher_hand_loss": 0.0,
        "teacher_hand_supported_fraction": 0.0,
        "teacher_move_source_loss": 0.0,
        "teacher_move_source_supported_fraction": 0.0,
        "teacher_attack_type_loss": 0.0,
        "teacher_action_loss": 0.0,
        "teacher_action_supported_fraction": 0.0,
        "teacher_action_margin_loss": 0.0,
        "teacher_action_margin_supported_fraction": 0.0,
        "teacher_action_margin_mean": 0.0,
        "teacher_action_margin_satisfied_fraction": 0.0,
        "teacher_same_family_action_margin_loss": 0.0,
        "teacher_same_family_action_margin_supported_fraction": 0.0,
        "teacher_same_family_action_margin_mean": 0.0,
        "teacher_same_family_action_margin_satisfied_fraction": 0.0,
        "teacher_same_family_action_loss": 0.0,
        "teacher_same_family_action_supported_fraction": 0.0,
        "teacher_public_heuristic_loss": 0.0,
        "teacher_public_heuristic_supported_fraction": 0.0,
        "teacher_public_heuristic_top1_mass": 0.0,
        "teacher_public_heuristic_target_entropy": 0.0,
        "teacher_public_nonpass_over_pass_loss": 0.0,
        "teacher_public_nonpass_over_pass_supported_fraction": 0.0,
        "teacher_public_nonpass_over_pass_margin_mean": 0.0,
        "teacher_public_nonpass_over_pass_satisfied_fraction": 0.0,
        "teacher_aux_loss": 0.0,
    }


def flatten_structured_teacher_labels(
    *,
    loss_mask: Tensor,
    teacher_family: Tensor,
    teacher_slot: Tensor,
    teacher_move_source: Tensor | None,
    teacher_attack_type: Tensor,
    teacher_action: Tensor | None,
    teacher_valid: Tensor,
) -> FlatStructuredTeacherLabels:
    return FlatStructuredTeacherLabels(
        loss_mask=loss_mask.reshape(-1).to(dtype=torch.float32),
        family=teacher_family.reshape(-1).to(dtype=torch.long),
        slot=teacher_slot.reshape(-1).to(dtype=torch.long),
        move_source=None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long),
        attack_type=teacher_attack_type.reshape(-1).to(dtype=torch.long),
        action=None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long),
        valid=teacher_valid.reshape(-1).to(dtype=torch.bool),
    )


def exact_action_family_rows(
    *,
    flat_teacher_family: Tensor,
    family_names: tuple[str, ...],
    exact_action_families: tuple[str, ...],
) -> Tensor | None:
    if not exact_action_families:
        return None
    exact_family_ids = resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=tuple(exact_action_families),
    )
    if not exact_family_ids:
        return None
    return torch.isin(
        flat_teacher_family,
        torch.as_tensor(exact_family_ids, device=flat_teacher_family.device, dtype=flat_teacher_family.dtype),
    )


def record_teacher_family_coverage(
    metrics: dict[str, float],
    *,
    active_rows: Tensor,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    play_family_id: int,
    move_family_id: int,
    attack_family_id: int,
) -> None:
    active_total = float(active_rows.float().sum().item())
    metrics["teacher_active_fraction"] = active_total / max(float(active_rows.numel()), 1.0)
    if active_total <= 0.0:
        return
    family_rows = active_rows & flat_teacher_valid & (flat_teacher_family >= 0)
    if play_family_id >= 0:
        metrics["teacher_main_play_character_fraction"] = float(
            ((family_rows & (flat_teacher_family == play_family_id)).float().sum().item()) / active_total
        )
    if move_family_id >= 0:
        metrics["teacher_main_move_fraction"] = float(
            ((family_rows & (flat_teacher_family == move_family_id)).float().sum().item()) / active_total
        )
    if attack_family_id >= 0:
        metrics["teacher_attack_fraction"] = float(
            ((family_rows & (flat_teacher_family == attack_family_id)).float().sum().item()) / active_total
        )


def weighted_structured_teacher_auxiliary_loss(
    *,
    terms: StructuredTeacherAuxiliaryLossTerms,
    coefs: StructuredTeacherAuxiliaryCoefficients,
) -> Tensor:
    total = (
        terms.family * float(coefs.family)
        + terms.slot * float(coefs.slot)
        + terms.attack_type * float(coefs.attack_type)
        + terms.action * float(coefs.action)
        + terms.same_family_action * float(coefs.same_family_action)
        + terms.action_margin * float(coefs.action_margin)
        + terms.same_family_action_margin * float(coefs.same_family_action_margin)
    )
    if terms.hand is not None:
        total = total + terms.hand * float(coefs.hand)
    if terms.move_source is not None:
        total = total + terms.move_source * float(coefs.move_source)
    if terms.public_heuristic is not None:
        total = total + terms.public_heuristic * float(coefs.public_heuristic)
    if terms.public_nonpass_over_pass is not None:
        total = total + terms.public_nonpass_over_pass * float(coefs.public_nonpass_over_pass)
    return total


def finalize_structured_teacher_auxiliary_loss(
    *,
    terms: StructuredTeacherAuxiliaryLossTerms,
    coefs: StructuredTeacherAuxiliaryCoefficients,
    metrics: dict[str, float],
    context: dict[str, Tensor] | None = None,
    value_dtype: torch.dtype | None = None,
) -> Tensor:
    total_aux = weighted_structured_teacher_auxiliary_loss(terms=terms, coefs=coefs)
    metrics["teacher_aux_loss"] = float(total_aux.detach().item())
    if context is not None:
        context["teacher_aux_loss"] = total_aux.detach()
    if value_dtype is not None:
        return total_aux.to(dtype=value_dtype)
    return total_aux


__all__ = [
    "FlatStructuredTeacherLabels",
    "StructuredTeacherAuxiliaryCoefficients",
    "StructuredTeacherAuxiliaryLossTerms",
    "exact_action_family_rows",
    "empty_structured_teacher_metrics",
    "finalize_structured_teacher_auxiliary_loss",
    "flatten_structured_teacher_labels",
    "record_teacher_family_coverage",
    "weighted_structured_teacher_auxiliary_loss",
]
