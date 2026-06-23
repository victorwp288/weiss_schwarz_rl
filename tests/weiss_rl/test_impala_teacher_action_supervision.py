from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher.factorized_actions import compute_factorized_teacher_action_supervision
from weiss_rl.learners.structured_teacher.packed import compute_packed_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.packed_actions import compute_packed_teacher_action_supervision

from .impala_test_support import (
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_compute_packed_teacher_action_supervision_matches_packed_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[5]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    direct = compute_packed_teacher_action_supervision(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_catalog=action_catalog,
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=1.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
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

    torch.testing.assert_close(packed_loss, direct.action_loss + direct.same_family_action_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_action_supervision_matches_factorized_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -3.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_log_probs = torch.log_softmax(family_logits, dim=-1)
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[0], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    same_family_logp = torch.tensor([[-0.1], [-0.4]], dtype=torch.float32)
    same_family_top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    zero = family_log_probs.sum() * 0.0

    direct = compute_factorized_teacher_action_supervision(
        family_log_probs=family_log_probs.reshape(-1, family_log_probs.shape[-1]),
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=zero,
        value_dtype=family_log_probs.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.3,
        same_family_action_coef=0.7,
        factorized_family_log_probs=family_log_probs,
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
    )
    expected_action_loss = direct.action_loss * 0.3 + direct.same_family_action_loss * 0.7

    torch.testing.assert_close(factorized_loss, expected_action_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)
