"""Dense-mask structured teacher-auxiliary loss branch."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.action_logp import masked_log_probs_and_entropy
from weiss_rl.learners.structured_auxiliary import (
    dense_group_log_probs,
    resolve_public_heuristic_family_ids,
    structured_group_lookup,
)
from weiss_rl.learners.structured_teacher.common import (
    StructuredTeacherAuxiliaryCoefficients,
    StructuredTeacherAuxiliaryLossTerms,
    empty_structured_teacher_metrics,
    exact_action_family_rows,
    finalize_structured_teacher_auxiliary_loss,
    flatten_structured_teacher_labels,
    record_teacher_family_coverage,
)
from weiss_rl.learners.structured_teacher.margin import (
    dense_teacher_action_margin_loss,
    dense_teacher_same_family_action_margin_loss,
)
from weiss_rl.learners.tensor_ops import weighted_mean


def compute_dense_structured_teacher_auxiliary_metrics(
    *,
    logits: Tensor,
    legal_mask: Tensor,
    teacher_family: Tensor,
    teacher_slot: Tensor,
    teacher_attack_type: Tensor,
    teacher_action: Tensor | None,
    teacher_valid: Tensor,
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
    zero: Tensor,
    public_heuristic_families: tuple[str, ...],
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    flat_labels = flatten_structured_teacher_labels(
        loss_mask=loss_mask,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_move_source=None,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
    )
    flat_loss_mask = flat_labels.loss_mask
    flat_teacher_family = flat_labels.family
    flat_teacher_slot = flat_labels.slot
    flat_teacher_attack_type = flat_labels.attack_type
    flat_teacher_action = flat_labels.action
    flat_teacher_valid = flat_labels.valid

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_legal_mask = legal_mask.reshape(-1, legal_mask.shape[-1]).to(dtype=torch.bool)
    masked_logits = torch.where(flat_legal_mask, flat_logits, torch.full_like(flat_logits, -1.0e9))

    lookup = structured_group_lookup(action_catalog, device=masked_logits.device)
    family_ids = lookup["family_ids"]
    play_slots = lookup["play_slots"]
    move_to_slots = lookup["move_to_slots"]
    attack_slots = lookup["attack_slots"]
    attack_types = lookup["attack_types"]
    family_index = lookup["family_index"]
    family_names = lookup["family_names"]
    attack_type_names = lookup["attack_type_names"]
    resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=tuple(public_heuristic_families),
    )
    exact_action_rows = exact_action_family_rows(
        flat_teacher_family=flat_teacher_family,
        family_names=family_names,
        exact_action_families=tuple(exact_action_families),
    )

    metrics = empty_structured_teacher_metrics()
    metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
    dense_context: dict[str, Tensor] = {}

    family_loss = zero
    family_rows = flat_teacher_valid & (flat_teacher_family >= 0)
    if bool(family_rows.any().item()):
        family_log_probs = dense_group_log_probs(
            masked_logits=masked_logits[family_rows],
            group_ids=family_ids,
            group_count=len(family_names),
        )
        valid_targets = flat_teacher_family[family_rows]
        row_weight = flat_loss_mask[family_rows]
        family_nll = -family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        family_loss = weighted_mean(family_nll, row_weight).to(dtype=logits.dtype)
        family_predictions = family_log_probs.argmax(dim=1)
        metrics["teacher_family_accuracy"] = float(
            ((family_predictions == valid_targets).float() * row_weight).sum().item()
            / max(float(row_weight.sum().item()), 1.0)
        )
        metrics["teacher_family_loss"] = float(family_loss.detach().item())
        dense_context["teacher_family_log_probs"] = family_log_probs.detach()

    dense_slot_loss_terms: list[Tensor] = []
    dense_slot_weight_terms: list[Tensor] = []
    slot_correct = 0.0
    slot_total = 0.0
    play_family_id = int(family_index.get("main_play_character", -1))
    move_family_id = int(family_index.get("main_move", -1))
    attack_family_id = int(family_index.get("attack", -1))
    record_teacher_family_coverage(
        metrics,
        active_rows=flat_loss_mask > 0.0,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        attack_family_id=attack_family_id,
    )

    play_rows = family_rows & (flat_teacher_family == play_family_id) & (flat_teacher_slot >= 0)
    if play_family_id >= 0 and bool(play_rows.any().item()):
        family_logits = masked_logits[play_rows]
        family_mask = flat_legal_mask[play_rows] & (family_ids == play_family_id).unsqueeze(0)
        group_log_probs = dense_group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=play_slots,
            group_count=max(int(action_catalog.max_stage), 1),
        )
        targets = flat_teacher_slot[play_rows]
        row_weight = flat_loss_mask[play_rows]
        dense_slot_loss_terms.append(-group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        dense_slot_weight_terms.append(row_weight)
        slot_predictions = group_log_probs.argmax(dim=1)
        play_slot_correct = float(((slot_predictions == targets).float() * row_weight).sum().item())
        play_slot_total = max(float(row_weight.sum().item()), 0.0)
        slot_correct += play_slot_correct
        slot_total += play_slot_total
        metrics["teacher_main_play_character_slot_accuracy"] = float(play_slot_correct / max(play_slot_total, 1.0))

    move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
    if move_family_id >= 0 and bool(move_rows.any().item()):
        family_logits = masked_logits[move_rows]
        family_mask = flat_legal_mask[move_rows] & (family_ids == move_family_id).unsqueeze(0)
        group_log_probs = dense_group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=move_to_slots,
            group_count=max(int(action_catalog.max_stage), 1),
        )
        targets = flat_teacher_slot[move_rows]
        row_weight = flat_loss_mask[move_rows]
        dense_slot_loss_terms.append(-group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        dense_slot_weight_terms.append(row_weight)
        slot_predictions = group_log_probs.argmax(dim=1)
        slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
        slot_total += max(float(row_weight.sum().item()), 0.0)

    attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
    if attack_family_id >= 0 and bool(attack_rows.any().item()):
        family_logits = masked_logits[attack_rows]
        family_mask = flat_legal_mask[attack_rows] & (family_ids == attack_family_id).unsqueeze(0)
        group_log_probs = dense_group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=attack_slots,
            group_count=max(int(action_catalog.attack_slot_count), 1),
        )
        targets = flat_teacher_slot[attack_rows]
        row_weight = flat_loss_mask[attack_rows]
        dense_slot_loss_terms.append(-group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        dense_slot_weight_terms.append(row_weight)
        slot_predictions = group_log_probs.argmax(dim=1)
        slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
        slot_total += max(float(row_weight.sum().item()), 0.0)

    slot_loss = zero
    if dense_slot_loss_terms:
        all_slot_losses = torch.cat(dense_slot_loss_terms, dim=0)
        all_slot_weights = torch.cat(dense_slot_weight_terms, dim=0)
        slot_loss = weighted_mean(all_slot_losses, all_slot_weights).to(dtype=logits.dtype)
        metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
        metrics["teacher_slot_loss"] = float(slot_loss.detach().item())

    attack_type_loss = zero
    attack_type_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
    if attack_family_id >= 0 and bool(attack_type_rows.any().item()) and attack_type_names:
        family_logits = masked_logits[attack_type_rows]
        family_mask = flat_legal_mask[attack_type_rows] & (family_ids == attack_family_id).unsqueeze(0)
        group_log_probs = dense_group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=attack_types,
            group_count=len(attack_type_names),
        )
        targets = flat_teacher_attack_type[attack_type_rows]
        row_weight = flat_loss_mask[attack_type_rows]
        attack_type_nll = -group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        attack_type_loss = weighted_mean(attack_type_nll, row_weight).to(dtype=logits.dtype)
        attack_type_predictions = group_log_probs.argmax(dim=1)
        metrics["teacher_attack_type_accuracy"] = float(
            ((attack_type_predictions == targets).float() * row_weight).sum().item()
            / max(float(row_weight.sum().item()), 1.0)
        )
        metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
        dense_context["teacher_attack_type_log_probs"] = group_log_probs.detach()

    action_loss = zero
    if flat_teacher_action is not None and float(action_coef) != 0.0:
        action_rows = flat_teacher_valid & (flat_teacher_action >= 0)
        if exact_action_rows is not None:
            action_rows = action_rows & exact_action_rows
        if bool(action_rows.any().item()):
            action_targets = flat_teacher_action[action_rows]
            action_weights = flat_loss_mask[action_rows]
            action_masks = flat_legal_mask[action_rows]
            action_logits = flat_logits[action_rows]
            action_log_probs = torch.full(
                action_targets.shape,
                float("-inf"),
                dtype=flat_logits.dtype,
                device=flat_logits.device,
            )
            predictions = torch.full_like(action_targets, -1)
            empty_rows = ~action_masks.any(dim=1)
            if bool((~empty_rows).any().item()):
                non_empty_targets = action_targets[~empty_rows]
                non_empty_masks = action_masks[~empty_rows]
                non_empty_logits = action_logits[~empty_rows]
                in_range = (non_empty_targets >= 0) & (non_empty_targets < non_empty_logits.shape[-1])
                if bool(in_range.any().item()):
                    selected_masks = non_empty_masks[in_range]
                    selected_targets = non_empty_targets[in_range]
                    supported = selected_masks.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
                    if bool(supported.any().item()):
                        supported_logits = non_empty_logits[in_range][supported]
                        supported_masks = selected_masks[supported]
                        supported_targets = selected_targets[supported]
                        supported_log_probs, _supported_entropy = masked_log_probs_and_entropy(
                            supported_logits,
                            supported_masks,
                        )
                        gather_log_probs = supported_log_probs.gather(1, supported_targets.unsqueeze(1)).squeeze(1)
                        action_log_probs[torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)[in_range][supported]] = (
                            gather_log_probs
                        )
                        predictions[torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)[in_range][supported]] = (
                            supported_log_probs.argmax(dim=1)
                        )
            if int(action_catalog.pass_action_id) >= 0 and bool(empty_rows.any().item()):
                pass_supported = action_targets[empty_rows] == int(action_catalog.pass_action_id)
                if bool(pass_supported.any().item()):
                    empty_indices = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)[pass_supported]
                    action_log_probs[empty_indices] = 0.0
                    predictions[empty_indices] = int(action_catalog.pass_action_id)
            supported_rows = torch.isfinite(action_log_probs)
            if float(action_weights.sum().item()) > 0.0:
                metrics["teacher_action_supported_fraction"] = float(
                    (action_weights[supported_rows].sum().item()) / max(float(action_weights.sum().item()), 1.0e-8)
                )
            if bool(supported_rows.any().item()):
                supported_weights = action_weights[supported_rows]
                supported_log_probs = action_log_probs[supported_rows]
                supported_predictions = predictions[supported_rows]
                supported_targets = action_targets[supported_rows]
                action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=logits.dtype)
                metrics["teacher_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_action_loss"] = float(action_loss.detach().item())
                dense_context["teacher_action_log_probs"] = supported_log_probs.detach()

    same_family_action_loss = zero
    if flat_teacher_action is not None and float(same_family_action_coef) != 0.0:
        same_family_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if exact_action_rows is not None:
            same_family_rows = same_family_rows & exact_action_rows
        if bool(same_family_rows.any().item()):
            row_targets = flat_teacher_action[same_family_rows]
            row_weights = flat_loss_mask[same_family_rows]
            row_logits = flat_logits[same_family_rows]
            row_masks = flat_legal_mask[same_family_rows]
            row_teacher_families = flat_teacher_family[same_family_rows]
            same_family_masks = row_masks & (family_ids.unsqueeze(0) == row_teacher_families.unsqueeze(1))
            same_family_log_probs, _same_family_entropy = masked_log_probs_and_entropy(
                row_logits,
                same_family_masks,
            )
            supported = same_family_masks.gather(1, row_targets.unsqueeze(1)).squeeze(1)
            if float(row_weights.sum().item()) > 0.0:
                metrics["teacher_same_family_action_supported_fraction"] = float(
                    (row_weights[supported].sum().item()) / max(float(row_weights.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_targets = row_targets[supported]
                supported_weights = row_weights[supported]
                supported_log_probs = (
                    same_family_log_probs[supported].gather(1, supported_targets.unsqueeze(1)).squeeze(1)
                )
                supported_predictions = same_family_log_probs[supported].argmax(dim=1)
                same_family_action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=logits.dtype)
                metrics["teacher_same_family_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                dense_context["teacher_same_family_action_log_probs"] = supported_log_probs.detach()
                supported_families = row_teacher_families[supported]
                main_play_supported = supported_families == play_family_id
                if bool(main_play_supported.any().item()):
                    play_weights = supported_weights[main_play_supported]
                    metrics["teacher_same_family_main_play_character_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_play_supported] == supported_targets[main_play_supported]
                            ).float()
                            * play_weights
                        )
                        .sum()
                        .item()
                        / max(float(play_weights.sum().item()), 1.0)
                    )
                main_move_family_id = int(family_index.get("main_move", -1))
                main_move_supported = supported_families == main_move_family_id
                if bool(main_move_supported.any().item()):
                    move_weights = supported_weights[main_move_supported]
                    metrics["teacher_same_family_main_move_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_move_supported] == supported_targets[main_move_supported]
                            ).float()
                            * move_weights
                        )
                        .sum()
                        .item()
                        / max(float(move_weights.sum().item()), 1.0)
                    )

    action_margin_loss = zero
    if flat_teacher_action is not None and float(action_margin_coef) != 0.0:
        action_margin_loss, action_margin_metrics, action_margin_context = dense_teacher_action_margin_loss(
            logits=logits,
            legal_mask=legal_mask,
            teacher_action=flat_teacher_action,
            teacher_valid=flat_teacher_valid if exact_action_rows is None else (flat_teacher_valid & exact_action_rows),
            loss_mask=flat_loss_mask,
            margin=float(action_margin),
            zero=zero,
        )
        metrics.update(action_margin_metrics)
        dense_context.update(action_margin_context)

    same_family_action_margin_loss = zero
    if flat_teacher_action is not None and float(same_family_action_margin_coef) != 0.0:
        same_family_action_margin_loss, same_family_margin_metrics, same_family_margin_context = (
            dense_teacher_same_family_action_margin_loss(
                logits=logits,
                legal_mask=legal_mask,
                action_family_ids=family_ids,
                teacher_action=flat_teacher_action,
                teacher_family=flat_teacher_family,
                teacher_valid=flat_teacher_valid
                if exact_action_rows is None
                else (flat_teacher_valid & exact_action_rows),
                loss_mask=flat_loss_mask,
                margin=float(same_family_action_margin),
                zero=zero,
            )
        )
        metrics.update(same_family_margin_metrics)
        dense_context.update(same_family_margin_context)

    total_aux = finalize_structured_teacher_auxiliary_loss(
        terms=StructuredTeacherAuxiliaryLossTerms(
            family=family_loss,
            slot=slot_loss,
            attack_type=attack_type_loss,
            action=action_loss,
            same_family_action=same_family_action_loss,
            action_margin=action_margin_loss,
            same_family_action_margin=same_family_action_margin_loss,
        ),
        coefs=StructuredTeacherAuxiliaryCoefficients(
            family=family_coef,
            slot=slot_coef,
            attack_type=attack_type_coef,
            action=action_coef,
            same_family_action=same_family_action_coef,
            action_margin=action_margin_coef,
            same_family_action_margin=same_family_action_margin_coef,
        ),
        metrics=metrics,
        context=dense_context,
        value_dtype=logits.dtype,
    )
    return total_aux, metrics, dense_context


__all__ = ["compute_dense_structured_teacher_auxiliary_metrics"]
