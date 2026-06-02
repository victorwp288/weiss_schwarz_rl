"""Packed family/slot/source/type teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView, packed_group_log_probs
from weiss_rl.learners.structured_teacher.common import record_teacher_family_coverage
from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True, slots=True)
class PackedTeacherGroupSupervisionResult:
    family_loss: Tensor
    slot_loss: Tensor
    move_source_loss: Tensor
    attack_type_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_packed_teacher_group_supervision(
    *,
    packed_view: PackedStructuredLegalView,
    flat_loss_mask: Tensor,
    flat_teacher_family: Tensor,
    flat_teacher_slot: Tensor,
    flat_teacher_move_source: Tensor | None,
    flat_teacher_attack_type: Tensor,
    flat_teacher_action: Tensor | None,
    flat_teacher_valid: Tensor,
    action_catalog: ActionCatalog,
    family_names: tuple[str, ...],
    family_index: dict[str, int],
    attack_type_names: tuple[str, ...],
    move_source_targets_by_action: Tensor | None,
    move_source_coef: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> PackedTeacherGroupSupervisionResult:
    metrics: dict[str, float] = {
        "teacher_valid_fraction": float(flat_teacher_valid.float().mean().item()),
    }
    context: dict[str, Tensor] = {}

    family_loss = zero
    family_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_family >= 0)
    family_log_probs = packed_group_log_probs(
        packed_view,
        group_ids=packed_view.family_ids,
        group_count=len(family_names),
    )
    if bool(family_rows.any().item()):
        valid_targets = flat_teacher_family[family_rows]
        row_weight = flat_loss_mask[family_rows]
        selected_family_log_probs = family_log_probs[family_rows]
        target_log_probs = selected_family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        supported = torch.isfinite(target_log_probs)
        if bool(supported.any().item()):
            valid_targets = valid_targets[supported]
            row_weight = row_weight[supported]
            selected_family_log_probs = selected_family_log_probs[supported]
            family_nll = -target_log_probs[supported]
            family_loss = weighted_mean(family_nll, row_weight).to(dtype=value_dtype)
            family_predictions = selected_family_log_probs.argmax(dim=1)
            metrics["teacher_family_accuracy"] = float(
                ((family_predictions == valid_targets).float() * row_weight).sum().item()
                / max(float(row_weight.sum().item()), 1.0)
            )
            metrics["teacher_family_loss"] = float(family_loss.detach().item())
            context["teacher_family_log_probs"] = selected_family_log_probs.detach()

    packed_slot_loss_terms: list[Tensor] = []
    packed_slot_weight_terms: list[Tensor] = []
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
        group_log_probs = packed_group_log_probs(
            packed_view,
            group_ids=packed_view.arg1,
            group_count=max(int(action_catalog.max_stage), 1),
            candidate_mask=packed_view.family_ids == play_family_id,
        )
        targets = flat_teacher_slot[play_rows]
        row_weight = flat_loss_mask[play_rows]
        selected_group_log_probs = group_log_probs[play_rows]
        target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        supported = torch.isfinite(target_log_probs)
        if bool(supported.any().item()):
            targets = targets[supported]
            row_weight = row_weight[supported]
            selected_group_log_probs = selected_group_log_probs[supported]
            packed_slot_loss_terms.append(-target_log_probs[supported])
            packed_slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            play_slot_correct = float(((slot_predictions == targets).float() * row_weight).sum().item())
            play_slot_total = max(float(row_weight.sum().item()), 0.0)
            slot_correct += play_slot_correct
            slot_total += play_slot_total
            metrics["teacher_main_play_character_slot_accuracy"] = float(play_slot_correct / max(play_slot_total, 1.0))

    move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
    if move_family_id >= 0 and bool(move_rows.any().item()):
        group_log_probs = packed_group_log_probs(
            packed_view,
            group_ids=packed_view.arg1,
            group_count=max(int(action_catalog.max_stage), 1),
            candidate_mask=packed_view.family_ids == move_family_id,
        )
        targets = flat_teacher_slot[move_rows]
        row_weight = flat_loss_mask[move_rows]
        selected_group_log_probs = group_log_probs[move_rows]
        target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        supported = torch.isfinite(target_log_probs)
        if bool(supported.any().item()):
            targets = targets[supported]
            row_weight = row_weight[supported]
            selected_group_log_probs = selected_group_log_probs[supported]
            packed_slot_loss_terms.append(-target_log_probs[supported])
            packed_slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
            slot_total += max(float(row_weight.sum().item()), 0.0)

    move_source_loss = zero
    if move_family_id >= 0 and float(move_source_coef) != 0.0:
        if flat_teacher_move_source is not None:
            move_source_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_move_source >= 0)
        elif flat_teacher_action is not None:
            move_source_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_action >= 0)
        else:
            move_source_rows = None
        if move_source_rows is not None and not bool(move_source_rows.any().item()):
            move_source_rows = None
    else:
        move_source_rows = None
    if move_source_rows is not None:
        group_log_probs = packed_group_log_probs(
            packed_view,
            group_ids=packed_view.arg0,
            group_count=max(int(action_catalog.max_stage), 1),
            candidate_mask=packed_view.family_ids == move_family_id,
        )
        if flat_teacher_move_source is not None:
            move_source_targets = flat_teacher_move_source[move_source_rows]
        else:
            assert flat_teacher_action is not None
            assert move_source_targets_by_action is not None
            move_source_targets = move_source_targets_by_action.index_select(0, flat_teacher_action[move_source_rows])
        valid_targets = move_source_targets >= 0
        if bool(valid_targets.any().item()):
            row_weight = flat_loss_mask[move_source_rows][valid_targets]
            selected_group_log_probs = group_log_probs[move_source_rows][valid_targets]
            move_source_targets = move_source_targets[valid_targets]
            target_log_probs = selected_group_log_probs.gather(1, move_source_targets.unsqueeze(1)).squeeze(1)
            supported = torch.isfinite(target_log_probs)
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_move_source_supported_fraction"] = float(
                    (row_weight[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                row_weight = row_weight[supported]
                move_source_targets = move_source_targets[supported]
                selected_group_log_probs = selected_group_log_probs[supported]
                target_log_probs = target_log_probs[supported]
                move_source_loss = weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
                move_source_predictions = selected_group_log_probs.argmax(dim=1)
                metrics["teacher_move_source_accuracy"] = float(
                    ((move_source_predictions == move_source_targets).float() * row_weight).sum().item()
                    / max(float(row_weight.sum().item()), 1.0)
                )
                metrics["teacher_move_source_loss"] = float(move_source_loss.detach().item())

    attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
    if attack_family_id >= 0 and bool(attack_rows.any().item()):
        group_log_probs = packed_group_log_probs(
            packed_view,
            group_ids=packed_view.arg0,
            group_count=max(int(action_catalog.attack_slot_count), 1),
            candidate_mask=packed_view.family_ids == attack_family_id,
        )
        targets = flat_teacher_slot[attack_rows]
        row_weight = flat_loss_mask[attack_rows]
        selected_group_log_probs = group_log_probs[attack_rows]
        target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        supported = torch.isfinite(target_log_probs)
        if bool(supported.any().item()):
            targets = targets[supported]
            row_weight = row_weight[supported]
            selected_group_log_probs = selected_group_log_probs[supported]
            packed_slot_loss_terms.append(-target_log_probs[supported])
            packed_slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
            slot_total += max(float(row_weight.sum().item()), 0.0)

    slot_loss = zero
    if packed_slot_loss_terms:
        all_slot_losses = torch.cat(packed_slot_loss_terms, dim=0)
        all_slot_weights = torch.cat(packed_slot_weight_terms, dim=0)
        slot_loss = weighted_mean(all_slot_losses, all_slot_weights).to(dtype=value_dtype)
        metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
        metrics["teacher_slot_loss"] = float(slot_loss.detach().item())

    attack_type_loss = zero
    attack_type_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
    if attack_family_id >= 0 and bool(attack_type_rows.any().item()) and attack_type_names:
        group_log_probs = packed_group_log_probs(
            packed_view,
            group_ids=packed_view.arg1,
            group_count=len(attack_type_names),
            candidate_mask=packed_view.family_ids == attack_family_id,
        )
        targets = flat_teacher_attack_type[attack_type_rows]
        row_weight = flat_loss_mask[attack_type_rows]
        selected_group_log_probs = group_log_probs[attack_type_rows]
        target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        supported = torch.isfinite(target_log_probs)
        if bool(supported.any().item()):
            targets = targets[supported]
            row_weight = row_weight[supported]
            selected_group_log_probs = selected_group_log_probs[supported]
            attack_type_nll = -target_log_probs[supported]
            attack_type_loss = weighted_mean(attack_type_nll, row_weight).to(dtype=value_dtype)
            attack_type_predictions = selected_group_log_probs.argmax(dim=1)
            metrics["teacher_attack_type_accuracy"] = float(
                ((attack_type_predictions == targets).float() * row_weight).sum().item()
                / max(float(row_weight.sum().item()), 1.0)
            )
            metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
            context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()

    return PackedTeacherGroupSupervisionResult(
        family_loss=family_loss,
        slot_loss=slot_loss,
        move_source_loss=move_source_loss,
        attack_type_loss=attack_type_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["PackedTeacherGroupSupervisionResult", "compute_packed_teacher_group_supervision"]
