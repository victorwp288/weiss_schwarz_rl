"""Packed structured teacher-auxiliary loss branch."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_teacher_packed_actions import compute_packed_teacher_action_supervision
from weiss_rl.learners.structured_teacher_packed_groups import compute_packed_teacher_group_supervision
from weiss_rl.learners.structured_teacher_packed_margins import compute_packed_teacher_margin_supervision
from weiss_rl.learners.structured_teacher_packed_public import compute_packed_teacher_public_supervision


def _exact_action_family_rows(
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


def compute_packed_structured_teacher_auxiliary_metrics(
    *,
    packed_view: PackedStructuredLegalView,
    packed_offsets: Tensor | None,
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
    zero: Tensor,
    value_dtype: torch.dtype,
    empty_metrics: dict[str, float],
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
    flat_teacher_move_source = (
        None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long)
    )
    flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
    flat_teacher_action = None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)

    catalog_metadata = structured_catalog_metadata(action_catalog)
    family_names = catalog_metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    public_heuristic_family_ids = resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=tuple(public_heuristic_families),
    )
    exact_action_family_rows = _exact_action_family_rows(
        flat_teacher_family=flat_teacher_family,
        family_names=family_names,
        exact_action_families=tuple(exact_action_families),
    )
    attack_type_names = catalog_metadata.attack_type_names
    move_source_targets_by_action = None
    if flat_teacher_move_source is None:
        move_source_targets_by_action = torch.as_tensor(
            catalog_metadata.move_from_slots,
            device=packed_view.logits.device,
            dtype=torch.long,
        )
    metrics = dict(empty_metrics)
    packed_context: dict[str, Tensor] = {}

    play_family_id = int(family_index.get("main_play_character", -1))
    move_family_id = int(family_index.get("main_move", -1))
    group_supervision = compute_packed_teacher_group_supervision(
        packed_view=packed_view,
        flat_loss_mask=flat_loss_mask,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_slot=flat_teacher_slot,
        flat_teacher_move_source=flat_teacher_move_source,
        flat_teacher_attack_type=flat_teacher_attack_type,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_valid=flat_teacher_valid,
        action_catalog=action_catalog,
        family_names=family_names,
        family_index=family_index,
        attack_type_names=attack_type_names,
        move_source_targets_by_action=move_source_targets_by_action,
        move_source_coef=move_source_coef,
        zero=zero,
        value_dtype=value_dtype,
    )
    family_loss = group_supervision.family_loss
    slot_loss = group_supervision.slot_loss
    move_source_loss = group_supervision.move_source_loss
    attack_type_loss = group_supervision.attack_type_loss
    metrics.update(group_supervision.metrics)
    packed_context.update(group_supervision.context)

    action_supervision = compute_packed_teacher_action_supervision(
        packed_view=packed_view,
        packed_offsets=packed_offsets,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        flat_loss_mask=flat_loss_mask,
        exact_action_family_rows=exact_action_family_rows,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        action_catalog=action_catalog,
        action_coef=action_coef,
        same_family_action_coef=same_family_action_coef,
        zero=zero,
        value_dtype=value_dtype,
    )
    action_loss = action_supervision.action_loss
    same_family_action_loss = action_supervision.same_family_action_loss
    metrics.update(action_supervision.metrics)
    packed_context.update(action_supervision.context)

    margin_supervision = compute_packed_teacher_margin_supervision(
        packed_view=packed_view,
        flat_teacher_action=flat_teacher_action,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        flat_loss_mask=flat_loss_mask,
        exact_action_family_rows=exact_action_family_rows,
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
    packed_context.update(margin_supervision.context)

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
    packed_context.update(public_supervision.context)

    total_aux = (
        family_loss * float(family_coef)
        + slot_loss * float(slot_coef)
        + move_source_loss * float(move_source_coef)
        + attack_type_loss * float(attack_type_coef)
        + action_loss * float(action_coef)
        + same_family_action_loss * float(same_family_action_coef)
        + action_margin_loss * float(action_margin_coef)
        + same_family_action_margin_loss * float(same_family_action_margin_coef)
        + public_heuristic_loss * float(public_heuristic_coef)
        + public_nonpass_over_pass_loss * float(public_nonpass_over_pass_coef)
    )
    metrics["teacher_aux_loss"] = float(total_aux.detach().item())
    packed_context["teacher_aux_loss"] = total_aux.detach()
    return total_aux.to(dtype=value_dtype), metrics, packed_context


__all__ = ["compute_packed_structured_teacher_auxiliary_metrics"]
