"""Typed IMPALA teacher-auxiliary input resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryLabels:
    family: Tensor | None
    slot: Tensor | None
    move_source: Tensor | None
    attack_type: Tensor | None
    action: Tensor | None
    valid: Tensor | None


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryCoefficients:
    family: float
    slot: float
    hand: float
    move_source: float
    attack_type: float
    action: float
    same_family_action: float
    action_margin: float
    action_margin_value: float
    same_family_action_margin: float
    same_family_action_margin_value: float
    exact_action_families: tuple[str, ...]
    public_heuristic: float
    public_heuristic_temperature: float
    public_nonpass_over_pass: float
    public_nonpass_over_pass_margin: float
    public_heuristic_families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryPackedInputs:
    ids: Tensor | None
    offsets: Tensor | None
    meta: Tensor | None
    view: Any


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryFactorizedInputs:
    family_log_probs: Tensor | None
    play_slot_log_probs: Tensor | None
    move_source_log_probs: Tensor | None
    move_slot_log_probs: Tensor | None
    attack_slot_log_probs: Tensor | None
    attack_type_log_probs: Tensor | None
    top_action_ids: Tensor | None
    same_family_action_logp: Tensor | None
    same_family_top_action_ids: Tensor | None
    same_family_arg0_logp: Tensor | None
    same_family_top_arg0: Tensor | None


@dataclass(frozen=True, slots=True)
class ImpalaTeacherAuxiliaryInputs:
    labels: ImpalaTeacherAuxiliaryLabels
    coefficients: ImpalaTeacherAuxiliaryCoefficients
    packed: ImpalaTeacherAuxiliaryPackedInputs
    factorized: ImpalaTeacherAuxiliaryFactorizedInputs


def resolve_impala_teacher_auxiliary_labels(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
    expected_shape: torch.Size,
) -> ImpalaTeacherAuxiliaryLabels:
    return ImpalaTeacherAuxiliaryLabels(
        family=learner._optional_time_major_index_field(
            batch_value(batch, "teacher_family"),
            field_name="teacher_family",
            expected_shape=expected_shape,
        ),
        slot=learner._optional_time_major_index_field(
            batch_value(batch, "teacher_slot"),
            field_name="teacher_slot",
            expected_shape=expected_shape,
        ),
        move_source=learner._optional_time_major_index_field(
            batch_value(batch, "teacher_move_source"),
            field_name="teacher_move_source",
            expected_shape=expected_shape,
        ),
        attack_type=learner._optional_time_major_index_field(
            batch_value(batch, "teacher_attack_type"),
            field_name="teacher_attack_type",
            expected_shape=expected_shape,
        ),
        action=learner._optional_time_major_index_field(
            batch_value(batch, "teacher_action"),
            field_name="teacher_action",
            expected_shape=expected_shape,
        ),
        valid=learner._optional_time_major_bool_field(
            batch_value(batch, "teacher_valid"),
            field_name="teacher_valid",
            expected_shape=expected_shape,
        ),
    )


def resolve_impala_teacher_auxiliary_coefficients(learner: Any) -> ImpalaTeacherAuxiliaryCoefficients:
    return ImpalaTeacherAuxiliaryCoefficients(
        family=float(learner.teacher_family_coef),
        slot=float(learner.teacher_slot_coef),
        hand=float(learner.teacher_hand_coef),
        move_source=float(learner.teacher_move_source_coef),
        attack_type=float(learner.teacher_attack_type_coef),
        action=float(learner.teacher_action_coef),
        same_family_action=float(learner.teacher_same_family_action_coef),
        action_margin=float(learner.teacher_action_margin_coef),
        action_margin_value=float(learner.teacher_action_margin),
        same_family_action_margin=float(learner.teacher_same_family_action_margin_coef),
        same_family_action_margin_value=float(learner.teacher_same_family_action_margin),
        exact_action_families=tuple(learner.teacher_exact_action_families),
        public_heuristic=float(learner.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(learner.teacher_public_heuristic_temperature),
        public_nonpass_over_pass=float(learner.teacher_public_nonpass_over_pass_coef),
        public_nonpass_over_pass_margin=float(learner.teacher_public_nonpass_over_pass_margin),
        public_heuristic_families=tuple(learner.teacher_public_heuristic_families),
    )


def resolve_impala_teacher_auxiliary_packed_inputs(
    *,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    packed_view: Any,
) -> ImpalaTeacherAuxiliaryPackedInputs:
    if packed_legal is None:
        return ImpalaTeacherAuxiliaryPackedInputs(
            ids=None,
            offsets=None,
            meta=None,
            view=packed_view,
        )
    return ImpalaTeacherAuxiliaryPackedInputs(
        ids=packed_legal[0],
        offsets=packed_legal[1],
        meta=packed_legal[2],
        view=packed_view,
    )


def resolve_impala_teacher_auxiliary_factorized_inputs(
    factorized_result: Any,
) -> ImpalaTeacherAuxiliaryFactorizedInputs:
    if factorized_result is None:
        return ImpalaTeacherAuxiliaryFactorizedInputs(
            family_log_probs=None,
            play_slot_log_probs=None,
            move_source_log_probs=None,
            move_slot_log_probs=None,
            attack_slot_log_probs=None,
            attack_type_log_probs=None,
            top_action_ids=None,
            same_family_action_logp=None,
            same_family_top_action_ids=None,
            same_family_arg0_logp=None,
            same_family_top_arg0=None,
        )
    return ImpalaTeacherAuxiliaryFactorizedInputs(
        family_log_probs=factorized_result.family_log_probs,
        play_slot_log_probs=factorized_result.play_slot_log_probs,
        move_source_log_probs=getattr(factorized_result, "move_source_log_probs", None),
        move_slot_log_probs=factorized_result.move_slot_log_probs,
        attack_slot_log_probs=factorized_result.attack_slot_log_probs,
        attack_type_log_probs=factorized_result.attack_type_log_probs,
        top_action_ids=getattr(factorized_result, "top_action_ids", None),
        same_family_action_logp=getattr(factorized_result, "same_family_action_logp", None),
        same_family_top_action_ids=getattr(factorized_result, "same_family_top_action_ids", None),
        same_family_arg0_logp=getattr(factorized_result, "same_family_arg0_logp", None),
        same_family_top_arg0=getattr(factorized_result, "same_family_top_arg0", None),
    )


def resolve_impala_teacher_auxiliary_inputs(
    *,
    learner: Any,
    batch: Any,
    batch_value: BatchValueGetter,
    expected_shape: torch.Size,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    packed_view: Any,
    factorized_result: Any,
) -> ImpalaTeacherAuxiliaryInputs:
    return ImpalaTeacherAuxiliaryInputs(
        labels=resolve_impala_teacher_auxiliary_labels(
            learner=learner,
            batch=batch,
            batch_value=batch_value,
            expected_shape=expected_shape,
        ),
        coefficients=resolve_impala_teacher_auxiliary_coefficients(learner),
        packed=resolve_impala_teacher_auxiliary_packed_inputs(
            packed_legal=packed_legal,
            packed_view=packed_view,
        ),
        factorized=resolve_impala_teacher_auxiliary_factorized_inputs(factorized_result),
    )


__all__ = [
    "BatchValueGetter",
    "ImpalaTeacherAuxiliaryCoefficients",
    "ImpalaTeacherAuxiliaryFactorizedInputs",
    "ImpalaTeacherAuxiliaryInputs",
    "ImpalaTeacherAuxiliaryLabels",
    "ImpalaTeacherAuxiliaryPackedInputs",
    "resolve_impala_teacher_auxiliary_coefficients",
    "resolve_impala_teacher_auxiliary_factorized_inputs",
    "resolve_impala_teacher_auxiliary_inputs",
    "resolve_impala_teacher_auxiliary_labels",
    "resolve_impala_teacher_auxiliary_packed_inputs",
]
