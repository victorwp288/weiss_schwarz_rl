"""Structured teacher-auxiliary loss computation for IMPALA-style learners."""

from __future__ import annotations

from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.structured_teacher.dense import (
    compute_dense_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.dispatch import (
    StructuredTeacherBranch,
    StructuredTeacherDispatch,
    StructuredTeacherRequiredLabels,
    StructuredTeacherZeroContext,
    resolve_structured_teacher_branch,
    resolve_structured_teacher_dispatch,
    resolve_structured_teacher_required_labels,
    resolve_structured_teacher_zero_context,
)
from weiss_rl.learners.structured_teacher.factorized import (
    compute_factorized_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.packed import (
    compute_packed_structured_teacher_auxiliary_metrics,
)


def compute_structured_teacher_auxiliary_metrics(
    *,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    teacher_family: Tensor | None,
    teacher_slot: Tensor | None,
    teacher_attack_type: Tensor | None,
    teacher_action: Tensor | None,
    teacher_valid: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    family_coef: float,
    slot_coef: float,
    attack_type_coef: float,
    action_coef: float,
    same_family_action_coef: float,
    hand_coef: float = 0.0,
    action_margin_coef: float = 0.0,
    action_margin: float = 0.5,
    same_family_action_margin_coef: float = 0.0,
    same_family_action_margin: float = 0.5,
    exact_action_families: tuple[str, ...] = (),
    move_source_coef: float = 0.0,
    public_heuristic_coef: float = 0.0,
    public_heuristic_temperature: float = 32.0,
    public_nonpass_over_pass_coef: float = 0.0,
    public_nonpass_over_pass_margin: float = 0.5,
    public_heuristic_families: tuple[str, ...] = (),
    public_heuristic_target_logits: Tensor | None = None,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: PackedStructuredLegalView | None = None,
    factorized_family_log_probs: Tensor | None = None,
    factorized_play_slot_log_probs: Tensor | None = None,
    factorized_move_source_log_probs: Tensor | None = None,
    factorized_move_slot_log_probs: Tensor | None = None,
    factorized_attack_slot_log_probs: Tensor | None = None,
    factorized_attack_type_log_probs: Tensor | None = None,
    factorized_top_action_ids: Tensor | None = None,
    factorized_same_family_action_logp: Tensor | None = None,
    factorized_same_family_top_action_ids: Tensor | None = None,
    factorized_same_family_arg0_logp: Tensor | None = None,
    factorized_same_family_top_arg0: Tensor | None = None,
    teacher_move_source: Tensor | None = None,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=packed_view,
        factorized_family_log_probs=factorized_family_log_probs,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
    )
    zero_context = dispatch.zero_context
    labels = dispatch.labels
    if labels is None:
        return zero_context.zero, zero_context.empty_metrics, {}
    packed_view = dispatch.packed_view
    branch = dispatch.branch

    if branch.use_factorized:
        assert factorized_family_log_probs is not None
        return compute_factorized_structured_teacher_auxiliary_metrics(
            factorized_family_log_probs=factorized_family_log_probs,
            factorized_play_slot_log_probs=factorized_play_slot_log_probs,
            factorized_move_source_log_probs=factorized_move_source_log_probs,
            factorized_move_slot_log_probs=factorized_move_slot_log_probs,
            factorized_attack_slot_log_probs=factorized_attack_slot_log_probs,
            factorized_attack_type_log_probs=factorized_attack_type_log_probs,
            factorized_top_action_ids=factorized_top_action_ids,
            factorized_same_family_action_logp=factorized_same_family_action_logp,
            factorized_same_family_top_action_ids=factorized_same_family_top_action_ids,
            factorized_same_family_arg0_logp=factorized_same_family_arg0_logp,
            factorized_same_family_top_arg0=factorized_same_family_top_arg0,
            teacher_family=labels.family,
            teacher_slot=labels.slot,
            teacher_attack_type=labels.attack_type,
            teacher_action=teacher_action,
            teacher_valid=labels.valid,
            teacher_move_source=teacher_move_source,
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            family_coef=family_coef,
            slot_coef=slot_coef,
            attack_type_coef=attack_type_coef,
            action_coef=action_coef,
            same_family_action_coef=same_family_action_coef,
            hand_coef=hand_coef,
            action_margin_coef=action_margin_coef,
            action_margin=action_margin,
            same_family_action_margin_coef=same_family_action_margin_coef,
            same_family_action_margin=same_family_action_margin,
            exact_action_families=exact_action_families,
            move_source_coef=move_source_coef,
            public_heuristic_coef=public_heuristic_coef,
            public_heuristic_temperature=public_heuristic_temperature,
            public_nonpass_over_pass_coef=public_nonpass_over_pass_coef,
            public_nonpass_over_pass_margin=public_nonpass_over_pass_margin,
            public_heuristic_families=public_heuristic_families,
            public_heuristic_target_logits=public_heuristic_target_logits,
            packed_view=packed_view,
            zero=zero_context.zero,
            value_dtype=zero_context.value_dtype,
        )

    if branch.use_packed:
        assert packed_view is not None
        return compute_packed_structured_teacher_auxiliary_metrics(
            packed_view=packed_view,
            packed_offsets=packed_offsets,
            teacher_family=labels.family,
            teacher_slot=labels.slot,
            teacher_attack_type=labels.attack_type,
            teacher_action=teacher_action,
            teacher_valid=labels.valid,
            teacher_move_source=teacher_move_source,
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            family_coef=family_coef,
            slot_coef=slot_coef,
            attack_type_coef=attack_type_coef,
            action_coef=action_coef,
            same_family_action_coef=same_family_action_coef,
            action_margin_coef=action_margin_coef,
            action_margin=action_margin,
            same_family_action_margin_coef=same_family_action_margin_coef,
            same_family_action_margin=same_family_action_margin,
            exact_action_families=exact_action_families,
            move_source_coef=move_source_coef,
            public_heuristic_coef=public_heuristic_coef,
            public_heuristic_temperature=public_heuristic_temperature,
            public_nonpass_over_pass_coef=public_nonpass_over_pass_coef,
            public_nonpass_over_pass_margin=public_nonpass_over_pass_margin,
            public_heuristic_families=public_heuristic_families,
            public_heuristic_target_logits=public_heuristic_target_logits,
            zero=zero_context.zero,
            value_dtype=zero_context.value_dtype,
            empty_metrics=zero_context.empty_metrics,
        )

    if not branch.use_dense:
        return zero_context.zero, zero_context.empty_metrics, {}

    assert logits is not None
    assert legal_mask is not None
    return compute_dense_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=labels.family,
        teacher_slot=labels.slot,
        teacher_attack_type=labels.attack_type,
        teacher_action=teacher_action,
        teacher_valid=labels.valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=family_coef,
        slot_coef=slot_coef,
        attack_type_coef=attack_type_coef,
        action_coef=action_coef,
        same_family_action_coef=same_family_action_coef,
        action_margin_coef=action_margin_coef,
        action_margin=action_margin,
        same_family_action_margin_coef=same_family_action_margin_coef,
        same_family_action_margin=same_family_action_margin,
        exact_action_families=exact_action_families,
        zero=zero_context.zero,
        public_heuristic_families=public_heuristic_families,
    )


__all__ = [
    "StructuredTeacherBranch",
    "StructuredTeacherDispatch",
    "StructuredTeacherRequiredLabels",
    "StructuredTeacherZeroContext",
    "compute_structured_teacher_auxiliary_metrics",
    "resolve_structured_teacher_branch",
    "resolve_structured_teacher_dispatch",
    "resolve_structured_teacher_required_labels",
    "resolve_structured_teacher_zero_context",
]
