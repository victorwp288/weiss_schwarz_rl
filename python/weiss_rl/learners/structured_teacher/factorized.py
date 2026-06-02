"""Factorized structured teacher-auxiliary loss branch."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_teacher.common import (
    StructuredTeacherAuxiliaryCoefficients,
    StructuredTeacherAuxiliaryLossTerms,
    empty_structured_teacher_metrics,
    exact_action_family_rows,
    finalize_structured_teacher_auxiliary_loss,
    flatten_structured_teacher_labels,
)
from weiss_rl.learners.structured_teacher.factorized_actions import compute_factorized_teacher_action_supervision
from weiss_rl.learners.structured_teacher.factorized_groups import compute_factorized_teacher_group_supervision
from weiss_rl.learners.structured_teacher.factorized_hand import compute_factorized_teacher_hand_supervision
from weiss_rl.learners.structured_teacher.packed_margins import compute_packed_teacher_margin_supervision
from weiss_rl.learners.structured_teacher.packed_public import compute_packed_teacher_public_supervision


def compute_factorized_structured_teacher_auxiliary_metrics(
    *,
    factorized_family_log_probs: Tensor,
    factorized_play_slot_log_probs: Tensor | None,
    factorized_move_source_log_probs: Tensor | None,
    factorized_move_slot_log_probs: Tensor | None,
    factorized_attack_slot_log_probs: Tensor | None,
    factorized_attack_type_log_probs: Tensor | None,
    factorized_top_action_ids: Tensor | None,
    factorized_same_family_action_logp: Tensor | None,
    factorized_same_family_top_action_ids: Tensor | None,
    factorized_same_family_arg0_logp: Tensor | None,
    factorized_same_family_top_arg0: Tensor | None,
    teacher_family: Tensor,
    teacher_slot: Tensor,
    teacher_attack_type: Tensor,
    teacher_action: Tensor | None,
    teacher_valid: Tensor,
    teacher_move_source: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    family_coef: float,
    slot_coef: float,
    attack_type_coef: float,
    action_coef: float,
    same_family_action_coef: float,
    hand_coef: float,
    action_margin_coef: float,
    action_margin: float,
    same_family_action_margin_coef: float,
    same_family_action_margin: float,
    exact_action_families: tuple[str, ...],
    move_source_coef: float,
    public_heuristic_coef: float,
    public_heuristic_temperature: float,
    public_nonpass_over_pass_coef: float,
    public_nonpass_over_pass_margin: float,
    public_heuristic_families: tuple[str, ...],
    public_heuristic_target_logits: Tensor | None,
    packed_view: PackedStructuredLegalView | None,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    flat_labels = flatten_structured_teacher_labels(
        loss_mask=loss_mask,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_move_source=teacher_move_source,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
    )
    flat_loss_mask = flat_labels.loss_mask
    flat_teacher_family = flat_labels.family
    flat_teacher_slot = flat_labels.slot
    flat_teacher_move_source = flat_labels.move_source
    flat_teacher_attack_type = flat_labels.attack_type
    flat_teacher_action = flat_labels.action
    flat_teacher_valid = flat_labels.valid
    family_log_probs = factorized_family_log_probs.reshape(-1, factorized_family_log_probs.shape[-1]).to(
        dtype=value_dtype
    )
    play_slot_log_probs = (
        None
        if factorized_play_slot_log_probs is None
        else factorized_play_slot_log_probs.reshape(-1, factorized_play_slot_log_probs.shape[-1]).to(dtype=value_dtype)
    )
    move_source_log_probs = (
        None
        if factorized_move_source_log_probs is None
        else factorized_move_source_log_probs.reshape(-1, factorized_move_source_log_probs.shape[-1]).to(
            dtype=value_dtype
        )
    )
    move_slot_log_probs = (
        None
        if factorized_move_slot_log_probs is None
        else factorized_move_slot_log_probs.reshape(-1, factorized_move_slot_log_probs.shape[-1]).to(dtype=value_dtype)
    )
    attack_slot_log_probs = (
        None
        if factorized_attack_slot_log_probs is None
        else factorized_attack_slot_log_probs.reshape(-1, factorized_attack_slot_log_probs.shape[-1]).to(
            dtype=value_dtype
        )
    )
    attack_type_log_probs = (
        None
        if factorized_attack_type_log_probs is None
        else factorized_attack_type_log_probs.reshape(-1, factorized_attack_type_log_probs.shape[-1]).to(
            dtype=value_dtype
        )
    )
    catalog_metadata = structured_catalog_metadata(action_catalog)
    family_names = catalog_metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    public_heuristic_family_ids = resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=tuple(public_heuristic_families),
    )
    exact_action_rows = exact_action_family_rows(
        flat_teacher_family=flat_teacher_family,
        family_names=family_names,
        exact_action_families=tuple(exact_action_families),
    )
    attack_type_names = catalog_metadata.attack_type_names
    metrics = empty_structured_teacher_metrics()
    factorized_context: dict[str, Tensor] = {}
    play_family_id = int(family_index.get("main_play_character", -1))
    main_event_family_id = int(family_index.get("main_play_event", -1))
    clock_from_hand_family_id = int(family_index.get("clock_from_hand", -1))
    climax_play_family_id = int(family_index.get("climax_play", -1))
    mulligan_select_family_id = int(family_index.get("mulligan_select", -1))
    move_family_id = int(family_index.get("main_move", -1))
    attack_family_id = int(family_index.get("attack", -1))
    move_source_targets_by_action = (
        torch.as_tensor(
            catalog_metadata.move_from_slots,
            device=family_log_probs.device,
            dtype=torch.long,
        )
        if flat_teacher_move_source is None
        else None
    )
    group_supervision = compute_factorized_teacher_group_supervision(
        family_log_probs=family_log_probs,
        play_slot_log_probs=play_slot_log_probs,
        move_source_log_probs=move_source_log_probs,
        move_slot_log_probs=move_slot_log_probs,
        attack_slot_log_probs=attack_slot_log_probs,
        attack_type_log_probs=attack_type_log_probs,
        flat_loss_mask=flat_loss_mask,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_slot=flat_teacher_slot,
        flat_teacher_move_source=flat_teacher_move_source,
        flat_teacher_attack_type=flat_teacher_attack_type,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_valid=flat_teacher_valid,
        attack_type_names=attack_type_names,
        move_source_targets_by_action=move_source_targets_by_action,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        attack_family_id=attack_family_id,
        move_source_coef=move_source_coef,
        zero=zero,
        value_dtype=value_dtype,
    )
    family_loss = group_supervision.family_loss
    slot_loss = group_supervision.slot_loss
    move_source_loss = group_supervision.move_source_loss
    attack_type_loss = group_supervision.attack_type_loss
    metrics.update(group_supervision.metrics)
    factorized_context.update(group_supervision.context)

    hand_loss = zero
    hand_supervision = compute_factorized_teacher_hand_supervision(
        factorized_same_family_arg0_logp=factorized_same_family_arg0_logp,
        factorized_same_family_top_arg0=factorized_same_family_top_arg0,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        flat_loss_mask=flat_loss_mask,
        exact_action_family_rows=exact_action_rows,
        hand_targets_by_action=torch.as_tensor(
            catalog_metadata.hand_indices,
            device=family_log_probs.device,
            dtype=torch.long,
        ),
        hand_family_ids=tuple(
            family_id
            for family_id in (
                play_family_id,
                main_event_family_id,
                clock_from_hand_family_id,
                climax_play_family_id,
                mulligan_select_family_id,
            )
            if family_id >= 0
        ),
        play_family_id=play_family_id,
        clock_from_hand_family_id=clock_from_hand_family_id,
        hand_coef=hand_coef,
        zero=zero,
        value_dtype=value_dtype,
    )
    hand_loss = hand_supervision.hand_loss
    metrics.update(hand_supervision.metrics)

    action_loss = zero
    same_family_action_loss = zero
    action_supervision = compute_factorized_teacher_action_supervision(
        family_log_probs=family_log_probs,
        factorized_top_action_ids=factorized_top_action_ids,
        factorized_same_family_action_logp=factorized_same_family_action_logp,
        factorized_same_family_top_action_ids=factorized_same_family_top_action_ids,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        flat_loss_mask=flat_loss_mask,
        exact_action_family_rows=exact_action_rows,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        action_coef=action_coef,
        same_family_action_coef=same_family_action_coef,
        zero=zero,
        value_dtype=value_dtype,
    )
    action_loss = action_supervision.action_loss
    same_family_action_loss = action_supervision.same_family_action_loss
    metrics.update(action_supervision.metrics)
    factorized_context.update(action_supervision.context)

    action_margin_loss = zero
    same_family_action_margin_loss = zero
    if packed_view is None:
        if flat_teacher_action is not None and float(action_margin_coef) != 0.0:
            metrics["teacher_action_margin_supported_fraction"] = 0.0
        if flat_teacher_action is not None and float(same_family_action_margin_coef) != 0.0:
            metrics["teacher_same_family_action_margin_supported_fraction"] = 0.0
    else:
        margin_supervision = compute_packed_teacher_margin_supervision(
            packed_view=packed_view,
            flat_teacher_action=flat_teacher_action,
            flat_teacher_family=flat_teacher_family,
            flat_teacher_valid=flat_teacher_valid,
            flat_loss_mask=flat_loss_mask,
            exact_action_family_rows=exact_action_rows,
            action_margin_coef=action_margin_coef,
            action_margin=action_margin,
            same_family_action_margin_coef=same_family_action_margin_coef,
            same_family_action_margin=same_family_action_margin,
            zero=zero,
            value_dtype=value_dtype,
        )
        action_margin_loss = margin_supervision.action_margin_loss
        same_family_action_margin_loss = margin_supervision.same_family_action_margin_loss
        metrics.update(margin_supervision.metrics)
        factorized_context.update(margin_supervision.context)

    public_heuristic_loss = zero
    public_nonpass_over_pass_loss = zero
    if packed_view is not None:
        public_supervision = compute_packed_teacher_public_supervision(
            packed_view=packed_view,
            public_heuristic_target_logits=public_heuristic_target_logits,
            public_heuristic_family_ids=tuple(public_heuristic_family_ids),
            flat_teacher_family=flat_teacher_family,
            flat_teacher_valid=flat_teacher_valid,
            flat_loss_mask=flat_loss_mask,
            pass_action_id=int(action_catalog.pass_action_id),
            public_heuristic_coef=public_heuristic_coef,
            public_heuristic_temperature=public_heuristic_temperature,
            public_nonpass_over_pass_coef=public_nonpass_over_pass_coef,
            public_nonpass_over_pass_margin=public_nonpass_over_pass_margin,
            zero=zero,
            value_dtype=value_dtype,
        )
        public_heuristic_loss = public_supervision.public_heuristic_loss
        public_nonpass_over_pass_loss = public_supervision.public_nonpass_over_pass_loss
        metrics.update(public_supervision.metrics)
        factorized_context.update(public_supervision.context)

    total_aux = finalize_structured_teacher_auxiliary_loss(
        terms=StructuredTeacherAuxiliaryLossTerms(
            family=family_loss,
            slot=slot_loss,
            hand=hand_loss,
            move_source=move_source_loss,
            attack_type=attack_type_loss,
            action=action_loss,
            same_family_action=same_family_action_loss,
            action_margin=action_margin_loss,
            same_family_action_margin=same_family_action_margin_loss,
            public_heuristic=public_heuristic_loss,
            public_nonpass_over_pass=public_nonpass_over_pass_loss,
        ),
        coefs=StructuredTeacherAuxiliaryCoefficients(
            family=family_coef,
            slot=slot_coef,
            hand=hand_coef,
            move_source=move_source_coef,
            attack_type=attack_type_coef,
            action=action_coef,
            same_family_action=same_family_action_coef,
            action_margin=action_margin_coef,
            same_family_action_margin=same_family_action_margin_coef,
            public_heuristic=public_heuristic_coef,
            public_nonpass_over_pass=public_nonpass_over_pass_coef,
        ),
        metrics=metrics,
    )
    return total_aux, metrics, factorized_context


__all__ = ["compute_factorized_structured_teacher_auxiliary_metrics"]
