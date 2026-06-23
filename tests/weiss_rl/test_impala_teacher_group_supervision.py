from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)
from weiss_rl.learners.structured_auxiliary import structured_catalog_metadata
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher.factorized_groups import compute_factorized_teacher_group_supervision
from weiss_rl.learners.structured_teacher.packed import compute_packed_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.packed_groups import compute_packed_teacher_group_supervision

from .impala_test_support import (
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_compute_packed_teacher_group_supervision_matches_packed_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((3, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((3, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[1, 0, [10, 11, 12, action_catalog.pass_action_id]] = True
    logits[1, 0, 10] = -2.0
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = -1.0
    logits[1, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[2, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[2, 0, move_action] = 3.5
    logits[2, 0, competing_move_action] = -0.5
    logits[2, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["attack"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [0], [int(move_decoded.to_slot or 0)]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [attack_type_index["direct"]], [-1]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [11], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    teacher_move_source = torch.tensor([[-1], [-1], [int(move_decoded.from_slot or 0)]], dtype=torch.long)
    loss_mask = torch.ones((3, 1), dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)

    direct = compute_packed_teacher_group_supervision(
        packed_view=packed_view,
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=teacher_move_source.reshape(-1),
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        action_catalog=action_catalog,
        family_names=metadata.family_names,
        family_index={name: index for index, name in enumerate(metadata.family_names)},
        attack_type_names=metadata.attack_type_names,
        move_source_targets_by_action=None,
        move_source_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=teacher_move_source,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.5,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(packed_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_group_supervision_matches_factorized_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    row_count = 3
    family_logits = torch.full((row_count, 1, len(action_catalog.families)), -4.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_logits[2, 0, family_index["attack"]] = 3.0
    play_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    play_slot_logits[0, 0, 0] = 3.0
    move_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_slot_logits[1, 0, int(move_decoded.to_slot or 0)] = 3.0
    move_source_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_source_logits[1, 0, int(move_decoded.from_slot or 0)] = 3.0
    attack_slot_logits = torch.zeros((row_count, 1, int(action_catalog.attack_slot_count)), dtype=torch.float32)
    attack_type_logits = torch.full((row_count, 1, len(action_catalog.attack_type_names)), -4.0)
    attack_type_logits[2, 0, attack_type_index["direct"]] = 3.0
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]], [family_index["attack"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [int(move_decoded.to_slot or 0)], [0]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [-1], [attack_type_index["direct"]]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [move_action], [10]], dtype=torch.long)
    teacher_valid = torch.ones((row_count, 1), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5], [0.25]], dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)
    zero = family_logits.sum() * 0.0

    direct = compute_factorized_teacher_group_supervision(
        family_log_probs=torch.log_softmax(family_logits, dim=-1).reshape(row_count, -1),
        play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1).reshape(row_count, -1),
        move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1).reshape(row_count, -1),
        move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1).reshape(row_count, -1),
        attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1).reshape(row_count, -1),
        attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1).reshape(row_count, -1),
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=None,
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        attack_type_names=tuple(action_catalog.attack_type_names),
        move_source_targets_by_action=torch.as_tensor(metadata.move_from_slots, dtype=torch.long),
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        attack_family_id=family_index["attack"],
        move_source_coef=1.0,
        zero=zero,
        value_dtype=family_logits.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.5,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1),
        factorized_move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1),
        factorized_move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1),
        factorized_attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1),
        factorized_attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(factorized_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)
