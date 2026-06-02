"""Factorized family/slot/source/type teacher supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.learners.structured_teacher.common import record_teacher_family_coverage
from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True, slots=True)
class FactorizedTeacherGroupSupervisionResult:
    family_loss: Tensor
    slot_loss: Tensor
    move_source_loss: Tensor
    attack_type_loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Tensor]


def compute_factorized_teacher_group_supervision(
    *,
    family_log_probs: Tensor,
    play_slot_log_probs: Tensor | None,
    move_source_log_probs: Tensor | None,
    move_slot_log_probs: Tensor | None,
    attack_slot_log_probs: Tensor | None,
    attack_type_log_probs: Tensor | None,
    flat_loss_mask: Tensor,
    flat_teacher_family: Tensor,
    flat_teacher_slot: Tensor,
    flat_teacher_move_source: Tensor | None,
    flat_teacher_attack_type: Tensor,
    flat_teacher_action: Tensor | None,
    flat_teacher_valid: Tensor,
    attack_type_names: tuple[str, ...],
    move_source_targets_by_action: Tensor | None,
    play_family_id: int,
    move_family_id: int,
    attack_family_id: int,
    move_source_coef: float,
    zero: Tensor,
    value_dtype: torch.dtype,
) -> FactorizedTeacherGroupSupervisionResult:
    metrics: dict[str, float] = {
        "teacher_valid_fraction": float(flat_teacher_valid.float().mean().item()),
    }
    context: dict[str, Tensor] = {}
    active_rows = flat_loss_mask > 0.0
    family_rows = active_rows & flat_teacher_valid & (flat_teacher_family >= 0)

    family_loss = zero
    if bool(family_rows.any().item()):
        valid_targets = flat_teacher_family[family_rows]
        row_weight = flat_loss_mask[family_rows]
        selected_family_log_probs = family_log_probs[family_rows]
        target_log_probs = selected_family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        family_loss = weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
        family_predictions = selected_family_log_probs.argmax(dim=1)
        metrics["teacher_family_accuracy"] = float(
            ((family_predictions == valid_targets).float() * row_weight).sum().item()
            / max(float(row_weight.sum().item()), 1.0)
        )
        metrics["teacher_family_loss"] = float(family_loss.detach().item())
        context["teacher_family_log_probs"] = selected_family_log_probs.detach()

    record_teacher_family_coverage(
        metrics,
        active_rows=active_rows,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        attack_family_id=attack_family_id,
    )

    slot_loss_terms: list[Tensor] = []
    slot_weight_terms: list[Tensor] = []
    slot_correct = 0.0
    slot_total = 0.0

    if play_slot_log_probs is not None and play_family_id >= 0:
        play_rows = family_rows & (flat_teacher_family == play_family_id) & (flat_teacher_slot >= 0)
        if bool(play_rows.any().item()):
            targets = flat_teacher_slot[play_rows]
            row_weight = flat_loss_mask[play_rows]
            selected_group_log_probs = play_slot_log_probs[play_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            slot_loss_terms.append(-target_log_probs)
            slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            play_slot_correct = float(((slot_predictions == targets).float() * row_weight).sum().item())
            play_slot_total = max(float(row_weight.sum().item()), 0.0)
            slot_correct += play_slot_correct
            slot_total += play_slot_total
            metrics["teacher_main_play_character_slot_accuracy"] = float(play_slot_correct / max(play_slot_total, 1.0))

    if move_slot_log_probs is not None and move_family_id >= 0:
        move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
        if bool(move_rows.any().item()):
            targets = flat_teacher_slot[move_rows]
            row_weight = flat_loss_mask[move_rows]
            selected_group_log_probs = move_slot_log_probs[move_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            slot_loss_terms.append(-target_log_probs)
            slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
            slot_total += max(float(row_weight.sum().item()), 0.0)

    move_source_loss = zero
    if move_source_log_probs is not None and move_family_id >= 0 and float(move_source_coef) != 0.0:
        if flat_teacher_move_source is not None:
            move_source_rows = (
                active_rows
                & flat_teacher_valid
                & (flat_teacher_family == move_family_id)
                & (flat_teacher_move_source >= 0)
            )
        elif flat_teacher_action is not None:
            move_source_rows = (
                active_rows & flat_teacher_valid & (flat_teacher_family == move_family_id) & (flat_teacher_action >= 0)
            )
        else:
            move_source_rows = None
        if move_source_rows is not None and bool(move_source_rows.any().item()):
            if flat_teacher_move_source is not None:
                move_source_targets = flat_teacher_move_source[move_source_rows]
            else:
                assert move_source_targets_by_action is not None
                assert flat_teacher_action is not None
                move_source_targets = move_source_targets_by_action.index_select(
                    0,
                    flat_teacher_action[move_source_rows],
                )
            valid_targets = move_source_targets >= 0
            if bool(valid_targets.any().item()):
                row_weight = flat_loss_mask[move_source_rows][valid_targets]
                selected_group_log_probs = move_source_log_probs[move_source_rows][valid_targets]
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

    if attack_slot_log_probs is not None and attack_family_id >= 0:
        attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
        if bool(attack_rows.any().item()):
            targets = flat_teacher_slot[attack_rows]
            row_weight = flat_loss_mask[attack_rows]
            selected_group_log_probs = attack_slot_log_probs[attack_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            slot_loss_terms.append(-target_log_probs)
            slot_weight_terms.append(row_weight)
            slot_predictions = selected_group_log_probs.argmax(dim=1)
            slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
            slot_total += max(float(row_weight.sum().item()), 0.0)

    slot_loss = zero
    if slot_loss_terms:
        slot_loss = weighted_mean(torch.cat(slot_loss_terms, dim=0), torch.cat(slot_weight_terms, dim=0)).to(
            dtype=value_dtype
        )
        metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
        metrics["teacher_slot_loss"] = float(slot_loss.detach().item())

    attack_type_loss = zero
    if attack_type_log_probs is not None and attack_family_id >= 0 and attack_type_names:
        attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
        if bool(attack_rows.any().item()):
            targets = flat_teacher_attack_type[attack_rows]
            row_weight = flat_loss_mask[attack_rows]
            selected_group_log_probs = attack_type_log_probs[attack_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            attack_type_loss = weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
            attack_type_predictions = selected_group_log_probs.argmax(dim=1)
            metrics["teacher_attack_type_accuracy"] = float(
                ((attack_type_predictions == targets).float() * row_weight).sum().item()
                / max(float(row_weight.sum().item()), 1.0)
            )
            metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
            context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()

    return FactorizedTeacherGroupSupervisionResult(
        family_loss=family_loss,
        slot_loss=slot_loss,
        move_source_loss=move_source_loss,
        attack_type_loss=attack_type_loss,
        metrics=metrics,
        context=context,
    )


__all__ = ["FactorizedTeacherGroupSupervisionResult", "compute_factorized_teacher_group_supervision"]
