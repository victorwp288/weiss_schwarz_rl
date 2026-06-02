"""Structured-teacher call mapping for IMPALA teacher auxiliary."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala.teacher_auxiliary_inputs import ImpalaTeacherAuxiliaryInputs
from weiss_rl.learners.structured_teacher.auxiliary import compute_structured_teacher_auxiliary_metrics


def compute_structured_teacher_auxiliary_from_impala_inputs(
    *,
    inputs: ImpalaTeacherAuxiliaryInputs,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    public_heuristic_target_logits: Tensor | None,
) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
    labels = inputs.labels
    coefficients = inputs.coefficients
    packed_inputs = inputs.packed
    factorized_inputs = inputs.factorized
    return compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=labels.family,
        teacher_slot=labels.slot,
        teacher_move_source=labels.move_source,
        teacher_attack_type=labels.attack_type,
        teacher_action=labels.action,
        teacher_valid=labels.valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=coefficients.family,
        slot_coef=coefficients.slot,
        hand_coef=coefficients.hand,
        move_source_coef=coefficients.move_source,
        attack_type_coef=coefficients.attack_type,
        action_coef=coefficients.action,
        same_family_action_coef=coefficients.same_family_action,
        action_margin_coef=coefficients.action_margin,
        action_margin=coefficients.action_margin_value,
        same_family_action_margin_coef=coefficients.same_family_action_margin,
        same_family_action_margin=coefficients.same_family_action_margin_value,
        exact_action_families=coefficients.exact_action_families,
        public_heuristic_coef=coefficients.public_heuristic,
        public_heuristic_temperature=coefficients.public_heuristic_temperature,
        public_nonpass_over_pass_coef=coefficients.public_nonpass_over_pass,
        public_nonpass_over_pass_margin=coefficients.public_nonpass_over_pass_margin,
        public_heuristic_families=coefficients.public_heuristic_families,
        public_heuristic_target_logits=public_heuristic_target_logits,
        packed_ids=packed_inputs.ids,
        packed_offsets=packed_inputs.offsets,
        packed_meta=packed_inputs.meta,
        packed_view=packed_inputs.view,
        factorized_family_log_probs=factorized_inputs.family_log_probs,
        factorized_play_slot_log_probs=factorized_inputs.play_slot_log_probs,
        factorized_move_source_log_probs=factorized_inputs.move_source_log_probs,
        factorized_move_slot_log_probs=factorized_inputs.move_slot_log_probs,
        factorized_attack_slot_log_probs=factorized_inputs.attack_slot_log_probs,
        factorized_attack_type_log_probs=factorized_inputs.attack_type_log_probs,
        factorized_top_action_ids=factorized_inputs.top_action_ids,
        factorized_same_family_action_logp=factorized_inputs.same_family_action_logp,
        factorized_same_family_top_action_ids=factorized_inputs.same_family_top_action_ids,
        factorized_same_family_arg0_logp=factorized_inputs.same_family_arg0_logp,
        factorized_same_family_top_arg0=factorized_inputs.same_family_top_arg0,
    )


__all__ = ["compute_structured_teacher_auxiliary_from_impala_inputs"]
