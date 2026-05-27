"""Factorized structured teacher-auxiliary loss branch."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView,
    packed_soft_target_cross_entropy,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_teacher_common import (
    empty_structured_teacher_metrics,
    record_teacher_family_coverage,
)
from weiss_rl.learners.structured_teacher_margin import (
    packed_public_nonpass_over_pass_loss,
    packed_teacher_action_margin_loss,
    packed_teacher_same_family_action_margin_loss,
)
from weiss_rl.learners.tensor_ops import weighted_mean


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
    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
    flat_teacher_move_source = (
        None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long)
    )
    flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
    flat_teacher_action = None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
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
    exact_action_family_rows = None
    if exact_action_families:
        exact_family_ids = resolve_public_heuristic_family_ids(
            family_names=family_names,
            requested_families=tuple(exact_action_families),
        )
        if exact_family_ids:
            exact_action_family_rows = torch.isin(
                flat_teacher_family,
                torch.as_tensor(exact_family_ids, device=flat_teacher_family.device, dtype=flat_teacher_family.dtype),
            )
    attack_type_names = catalog_metadata.attack_type_names
    metrics = empty_structured_teacher_metrics()
    metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
    factorized_context: dict[str, Tensor] = {}
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
        factorized_context["teacher_family_log_probs"] = selected_family_log_probs.detach()
    slot_loss_terms: list[Tensor] = []
    slot_weight_terms: list[Tensor] = []
    slot_correct = 0.0
    slot_total = 0.0
    play_family_id = int(family_index.get("main_play_character", -1))
    main_event_family_id = int(family_index.get("main_play_event", -1))
    clock_from_hand_family_id = int(family_index.get("clock_from_hand", -1))
    climax_play_family_id = int(family_index.get("climax_play", -1))
    mulligan_select_family_id = int(family_index.get("mulligan_select", -1))
    move_family_id = int(family_index.get("main_move", -1))
    attack_family_id = int(family_index.get("attack", -1))
    record_teacher_family_coverage(
        metrics,
        active_rows=active_rows,
        flat_teacher_family=flat_teacher_family,
        flat_teacher_valid=flat_teacher_valid,
        play_family_id=play_family_id,
        move_family_id=move_family_id,
        attack_family_id=attack_family_id,
    )
    move_source_targets_by_action = None
    if flat_teacher_move_source is None:
        move_source_targets_by_action = torch.as_tensor(
            catalog_metadata.move_from_slots,
            device=family_log_probs.device,
            dtype=torch.long,
        )
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
    hand_loss = zero
    if (
        flat_teacher_action is not None
        and float(hand_coef) != 0.0
        and factorized_same_family_arg0_logp is not None
        and factorized_same_family_top_arg0 is not None
    ):
        hand_family_ids = tuple(
            family_id
            for family_id in (
                play_family_id,
                main_event_family_id,
                clock_from_hand_family_id,
                climax_play_family_id,
                mulligan_select_family_id,
            )
            if family_id >= 0
        )
        if hand_family_ids:
            hand_targets_by_action = torch.as_tensor(
                catalog_metadata.hand_indices,
                device=family_log_probs.device,
                dtype=torch.long,
            )
            valid_action_rows = (flat_teacher_action >= 0) & (
                flat_teacher_action < int(hand_targets_by_action.shape[0])
            )
            hand_targets = torch.full_like(flat_teacher_action, -1)
            if bool(valid_action_rows.any().item()):
                hand_targets[valid_action_rows] = hand_targets_by_action.index_select(
                    0,
                    flat_teacher_action[valid_action_rows],
                )
            hand_rows = flat_teacher_valid & (hand_targets >= 0)
            hand_rows = hand_rows & torch.isin(
                flat_teacher_family,
                torch.as_tensor(hand_family_ids, device=flat_teacher_family.device, dtype=flat_teacher_family.dtype),
            )
            if exact_action_family_rows is not None:
                hand_rows = hand_rows & exact_action_family_rows
            same_family_arg0_logp = factorized_same_family_arg0_logp.reshape(-1).to(dtype=value_dtype)
            same_family_top_arg0 = factorized_same_family_top_arg0.reshape(-1).to(dtype=torch.long)
            supported = hand_rows & torch.isfinite(same_family_arg0_logp)
            if bool(hand_rows.any().item()):
                row_weight = flat_loss_mask[hand_rows]
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_hand_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
            if bool(supported.any().item()):
                supported_weights = flat_loss_mask[supported]
                supported_targets = hand_targets[supported]
                supported_predictions = same_family_top_arg0[supported]
                hand_loss = weighted_mean(-same_family_arg0_logp[supported], supported_weights).to(dtype=value_dtype)
                metrics["teacher_hand_loss"] = float(hand_loss.detach().item())
                metrics["teacher_hand_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                supported_families = flat_teacher_family[supported]
                main_play_supported = supported_families == play_family_id
                if bool(main_play_supported.any().item()):
                    play_weights = supported_weights[main_play_supported]
                    metrics["teacher_main_play_character_hand_accuracy"] = float(
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
                clock_supported = supported_families == clock_from_hand_family_id
                if bool(clock_supported.any().item()):
                    clock_weights = supported_weights[clock_supported]
                    metrics["teacher_clock_from_hand_accuracy"] = float(
                        (
                            (supported_predictions[clock_supported] == supported_targets[clock_supported]).float()
                            * clock_weights
                        )
                        .sum()
                        .item()
                        / max(float(clock_weights.sum().item()), 1.0)
                    )
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
            factorized_context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()
    action_loss = zero
    if flat_teacher_action is not None and float(action_coef) != 0.0 and factorized_same_family_action_logp is not None:
        action_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if exact_action_family_rows is not None:
            action_rows = action_rows & exact_action_family_rows
        if bool(action_rows.any().item()):
            teacher_family_log_probs = family_log_probs.gather(
                1,
                torch.clamp(flat_teacher_family, min=0).unsqueeze(1),
            ).squeeze(1)
            teacher_action_log_probs = teacher_family_log_probs + factorized_same_family_action_logp.reshape(-1).to(
                dtype=value_dtype
            )
            row_weight = flat_loss_mask[action_rows]
            supported = action_rows & torch.isfinite(teacher_action_log_probs)
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_log_probs = teacher_action_log_probs[supported]
                supported_weights = flat_loss_mask[supported]
                action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                if factorized_top_action_ids is not None:
                    supported_predictions = factorized_top_action_ids.reshape(-1).to(dtype=torch.long)[supported]
                    supported_targets = flat_teacher_action[supported]
                    metrics["teacher_action_accuracy"] = float(
                        ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                metrics["teacher_action_loss"] = float(action_loss.detach().item())
                factorized_context["teacher_action_log_probs"] = supported_log_probs.detach()
    same_family_action_loss = zero
    if (
        flat_teacher_action is not None
        and float(same_family_action_coef) != 0.0
        and factorized_same_family_action_logp is not None
        and factorized_same_family_top_action_ids is not None
    ):
        same_family_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if exact_action_family_rows is not None:
            same_family_rows = same_family_rows & exact_action_family_rows
        if bool(same_family_rows.any().item()):
            same_family_log_probs = factorized_same_family_action_logp.reshape(-1).to(dtype=value_dtype)
            same_family_top_actions = factorized_same_family_top_action_ids.reshape(-1).to(dtype=torch.long)
            row_weight = flat_loss_mask[same_family_rows]
            supported = same_family_rows & torch.isfinite(same_family_log_probs)
            if float(row_weight.sum().item()) > 0.0:
                metrics["teacher_same_family_action_supported_fraction"] = float(
                    (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_log_probs = same_family_log_probs[supported]
                supported_weights = flat_loss_mask[supported]
                supported_predictions = same_family_top_actions[supported]
                supported_targets = flat_teacher_action[supported]
                same_family_action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                metrics["teacher_same_family_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                factorized_context["teacher_same_family_action_log_probs"] = supported_log_probs.detach()
                supported_families = flat_teacher_family[supported]
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
                main_move_supported = supported_families == move_family_id
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
        if packed_view is None:
            metrics["teacher_action_margin_supported_fraction"] = 0.0
        else:
            action_margin_loss, action_margin_metrics, action_margin_context = packed_teacher_action_margin_loss(
                packed_view=packed_view,
                teacher_action=flat_teacher_action,
                teacher_valid=flat_teacher_valid
                if exact_action_family_rows is None
                else (flat_teacher_valid & exact_action_family_rows),
                loss_mask=flat_loss_mask,
                margin=float(action_margin),
                zero=zero,
                value_dtype=value_dtype,
            )
            metrics.update(action_margin_metrics)
            factorized_context.update(action_margin_context)

    same_family_action_margin_loss = zero
    if flat_teacher_action is not None and float(same_family_action_margin_coef) != 0.0:
        if packed_view is None:
            metrics["teacher_same_family_action_margin_supported_fraction"] = 0.0
        else:
            same_family_action_margin_loss, same_family_margin_metrics, same_family_margin_context = (
                packed_teacher_same_family_action_margin_loss(
                    packed_view=packed_view,
                    teacher_action=flat_teacher_action,
                    teacher_family=flat_teacher_family,
                    teacher_valid=flat_teacher_valid
                    if exact_action_family_rows is None
                    else (flat_teacher_valid & exact_action_family_rows),
                    loss_mask=flat_loss_mask,
                    margin=float(same_family_action_margin),
                    zero=zero,
                    value_dtype=value_dtype,
                )
            )
            metrics.update(same_family_margin_metrics)
            factorized_context.update(same_family_margin_context)
    public_heuristic_loss = zero
    public_nonpass_over_pass_loss = zero
    if packed_view is not None and public_heuristic_target_logits is not None and float(public_heuristic_coef) != 0.0:
        public_rows = packed_view.row_has_candidates & flat_teacher_valid
        if public_heuristic_family_ids:
            public_rows = public_rows & torch.isin(
                flat_teacher_family,
                torch.as_tensor(
                    public_heuristic_family_ids,
                    device=flat_teacher_family.device,
                    dtype=flat_teacher_family.dtype,
                ),
            )
        if bool(public_rows.any().item()):
            row_cross_entropy, row_student_top_mass, row_target_entropy = packed_soft_target_cross_entropy(
                packed_view,
                target_logits=public_heuristic_target_logits,
                temperature=float(public_heuristic_temperature),
            )
            public_weights = flat_loss_mask[public_rows]
            if float(public_weights.sum().item()) > 0.0:
                metrics["teacher_public_heuristic_supported_fraction"] = 1.0
                metrics["teacher_public_heuristic_top1_mass"] = float(
                    weighted_mean(row_student_top_mass[public_rows], public_weights).item()
                )
                metrics["teacher_public_heuristic_target_entropy"] = float(
                    weighted_mean(row_target_entropy[public_rows], public_weights).item()
                )
                public_heuristic_loss = weighted_mean(
                    row_cross_entropy[public_rows],
                    public_weights,
                ).to(dtype=value_dtype)
                metrics["teacher_public_heuristic_loss"] = float(public_heuristic_loss.detach().item())
    if (
        packed_view is not None
        and public_heuristic_target_logits is not None
        and float(public_nonpass_over_pass_coef) != 0.0
    ):
        public_nonpass_over_pass_loss, public_nonpass_metrics, public_nonpass_context = (
            packed_public_nonpass_over_pass_loss(
                packed_view=packed_view,
                target_logits=public_heuristic_target_logits,
                pass_action_id=int(action_catalog.pass_action_id),
                teacher_valid=flat_teacher_valid,
                loss_mask=flat_loss_mask,
                margin=float(public_nonpass_over_pass_margin),
                zero=zero,
                value_dtype=value_dtype,
            )
        )
        metrics.update(public_nonpass_metrics)
        factorized_context.update(public_nonpass_context)

    total_aux = (
        family_loss * float(family_coef)
        + slot_loss * float(slot_coef)
        + hand_loss * float(hand_coef)
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
    return total_aux, metrics, factorized_context


__all__ = ["compute_factorized_structured_teacher_auxiliary_metrics"]
