"""Structured teacher-auxiliary loss computation for IMPALA-style learners."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.action_logp import packed_selected_action_logp, packed_subset_action_logp_and_top_action
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView,
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_teacher_common import (
    empty_structured_teacher_metrics,
    record_teacher_family_coverage,
)
from weiss_rl.learners.structured_teacher_dense import (
    compute_dense_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher_factorized import (
    compute_factorized_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher_margin import (
    packed_public_nonpass_over_pass_loss,
    packed_teacher_action_margin_loss,
    packed_teacher_same_family_action_margin_loss,
)
from weiss_rl.learners.tensor_ops import segment_max, weighted_mean


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
    zero_source = logits
    if zero_source is None and packed_view is not None:
        zero_source = packed_view.logits
    if zero_source is None:
        zero_source = loss_mask
    zero = zero_source.sum() * 0.0
    value_dtype = zero.dtype
    empty_metrics = empty_structured_teacher_metrics()

    if teacher_family is None or teacher_slot is None or teacher_attack_type is None or teacher_valid is None:
        return zero, empty_metrics, {}

    if factorized_family_log_probs is not None:
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
            teacher_family=teacher_family,
            teacher_slot=teacher_slot,
            teacher_attack_type=teacher_attack_type,
            teacher_action=teacher_action,
            teacher_valid=teacher_valid,
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
            zero=zero,
            value_dtype=value_dtype,
        )

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
    flat_teacher_move_source = (
        None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long)
    )
    flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
    flat_teacher_action = None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    packed_view = (
        packed_view
        if packed_view is not None
        else packed_structured_legal_view(
            logits=logits,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    )
    if packed_view is not None:
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
        metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
        packed_context: dict[str, Tensor] = {}

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
                packed_context["teacher_family_log_probs"] = selected_family_log_probs.detach()

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
                metrics["teacher_main_play_character_slot_accuracy"] = float(
                    play_slot_correct / max(play_slot_total, 1.0)
                )

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
                move_source_rows = (
                    family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_move_source >= 0)
                )
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
                move_source_targets = move_source_targets_by_action.index_select(
                    0, flat_teacher_action[move_source_rows]
                )
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
                packed_context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()

        action_loss = zero
        if flat_teacher_action is not None and float(action_coef) != 0.0:
            action_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_action >= 0)
            if exact_action_family_rows is not None:
                action_rows = action_rows & exact_action_family_rows
            if bool(action_rows.any().item()):
                teacher_action_log_probs = (
                    packed_selected_action_logp(
                        packed_view.logits,
                        packed_view.action_ids,
                        packed_offsets
                        if packed_offsets is not None
                        else packed_view.row_indices.new_zeros((packed_view.row_count + 1,)),
                        flat_teacher_action,
                        pass_action_id=int(action_catalog.pass_action_id),
                        strict=False,
                    )
                    .reshape(-1)
                    .to(dtype=value_dtype)
                )
                supported = action_rows & torch.isfinite(teacher_action_log_probs)
                row_weight = flat_loss_mask[action_rows]
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_log_probs = teacher_action_log_probs[supported]
                    supported_weights = flat_loss_mask[supported]
                    action_loss = weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                    top_logits = segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
                    top_matches = packed_view.logits >= (top_logits.index_select(0, packed_view.row_indices) - 1.0e-6)
                    top_action_ids = torch.full(
                        (packed_view.row_count,),
                        -1,
                        dtype=torch.long,
                        device=packed_view.logits.device,
                    )
                    top_action_ids.scatter_reduce_(
                        0,
                        packed_view.row_indices.to(dtype=torch.long),
                        torch.where(
                            top_matches,
                            packed_view.action_ids.to(dtype=torch.long),
                            torch.full_like(packed_view.action_ids, -1),
                        ),
                        reduce="amax",
                        include_self=True,
                    )
                    metrics["teacher_action_accuracy"] = float(
                        ((top_action_ids[supported] == flat_teacher_action[supported]).float() * supported_weights)
                        .sum()
                        .item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                    metrics["teacher_action_loss"] = float(action_loss.detach().item())
                    packed_context["teacher_action_log_probs"] = supported_log_probs.detach()

        same_family_action_loss = zero
        if flat_teacher_action is not None and float(same_family_action_coef) != 0.0:
            same_family_rows = (
                packed_view.row_has_candidates
                & flat_teacher_valid
                & (flat_teacher_action >= 0)
                & (flat_teacher_family >= 0)
            )
            if exact_action_family_rows is not None:
                same_family_rows = same_family_rows & exact_action_family_rows
            if bool(same_family_rows.any().item()):
                candidate_mask = packed_view.family_ids == flat_teacher_family.index_select(
                    0,
                    packed_view.row_indices.to(dtype=torch.long),
                )
                same_family_log_probs, same_family_top_actions = packed_subset_action_logp_and_top_action(
                    packed_view,
                    flat_teacher_action,
                    candidate_mask=candidate_mask,
                    strict=False,
                )
                same_family_log_probs = same_family_log_probs.reshape(-1).to(dtype=value_dtype)
                same_family_top_actions = same_family_top_actions.reshape(-1).to(dtype=torch.long)
                supported = same_family_rows & torch.isfinite(same_family_log_probs)
                row_weight = flat_loss_mask[same_family_rows]
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_same_family_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_weights = flat_loss_mask[supported]
                    supported_targets = flat_teacher_action[supported]
                    same_family_action_loss = weighted_mean(
                        -same_family_log_probs[supported],
                        supported_weights,
                    ).to(dtype=value_dtype)
                    metrics["teacher_same_family_action_accuracy"] = float(
                        ((same_family_top_actions[supported] == supported_targets).float() * supported_weights)
                        .sum()
                        .item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                    metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                    packed_context["teacher_same_family_action_log_probs"] = same_family_log_probs[supported].detach()
                    main_play_supported = supported & (flat_teacher_family == play_family_id)
                    if bool(main_play_supported.any().item()):
                        main_play_weights = flat_loss_mask[main_play_supported]
                        metrics["teacher_same_family_main_play_character_accuracy"] = float(
                            (
                                (
                                    same_family_top_actions[main_play_supported]
                                    == flat_teacher_action[main_play_supported]
                                ).float()
                                * main_play_weights
                            )
                            .sum()
                            .item()
                            / max(float(main_play_weights.sum().item()), 1.0)
                        )
                    main_move_family_id = int(family_index.get("main_move", -1))
                    main_move_supported = supported & (flat_teacher_family == main_move_family_id)
                    if bool(main_move_supported.any().item()):
                        main_move_weights = flat_loss_mask[main_move_supported]
                        metrics["teacher_same_family_main_move_accuracy"] = float(
                            (
                                (
                                    same_family_top_actions[main_move_supported]
                                    == flat_teacher_action[main_move_supported]
                                ).float()
                                * main_move_weights
                            )
                            .sum()
                            .item()
                            / max(float(main_move_weights.sum().item()), 1.0)
                        )

        action_margin_loss = zero
        if flat_teacher_action is not None and float(action_margin_coef) != 0.0:
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
            packed_context.update(action_margin_context)

        same_family_action_margin_loss = zero
        if flat_teacher_action is not None and float(same_family_action_margin_coef) != 0.0:
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
            packed_context.update(same_family_margin_context)

        public_heuristic_loss = zero
        public_nonpass_over_pass_loss = zero
        if public_heuristic_target_logits is not None and float(public_heuristic_coef) != 0.0:
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
        if public_heuristic_target_logits is not None and float(public_nonpass_over_pass_coef) != 0.0:
            (
                public_nonpass_over_pass_loss,
                public_nonpass_over_pass_metrics,
                public_nonpass_over_pass_context,
            ) = packed_public_nonpass_over_pass_loss(
                packed_view=packed_view,
                target_logits=public_heuristic_target_logits,
                pass_action_id=int(action_catalog.pass_action_id),
                teacher_valid=flat_teacher_valid,
                loss_mask=flat_loss_mask,
                margin=float(public_nonpass_over_pass_margin),
                zero=zero,
                value_dtype=value_dtype,
            )
            metrics.update(public_nonpass_over_pass_metrics)
            packed_context.update(public_nonpass_over_pass_context)

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

    if logits is None or legal_mask is None:
        return zero, empty_metrics, {}

    return compute_dense_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
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
        zero=zero,
        public_heuristic_families=public_heuristic_families,
    )


__all__ = ["compute_structured_teacher_auxiliary_metrics"]
