from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)

from .impala_test_support import (
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
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
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[0, 0, move_action] = 4.0
    logits[0, 0, competing_move_action] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_skips_unsupported_packed_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1 carries attack teacher labels but only exposes pass legally, which previously
    # produced NaNs in the packed grouped-log-prob path.
    legal_mask[1, 0, [19]] = True
    logits[1, 0, 19] = 1.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    packed_loss, packed_metrics, packed_context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    assert torch.isfinite(packed_loss)
    assert np.isfinite(packed_metrics["teacher_aux_loss"])
    assert np.isfinite(packed_metrics["teacher_family_loss"])
    assert np.isfinite(packed_metrics["teacher_slot_loss"])
    assert np.isfinite(packed_metrics["teacher_attack_type_loss"])
    assert np.isfinite(packed_metrics["teacher_action_loss"])
    assert np.isfinite(packed_metrics["teacher_same_family_action_loss"])
    assert packed_metrics["teacher_action_supported_fraction"] == pytest.approx(0.5)
    assert packed_metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(0.5)
    assert "teacher_attack_type_log_probs" not in packed_context
    assert "teacher_family_log_probs" in packed_context
    assert "teacher_action_log_probs" in packed_context
    assert "teacher_same_family_action_log_probs" in packed_context
    assert not torch.isnan(packed_context["teacher_family_log_probs"]).any()


def test_compute_structured_teacher_auxiliary_metrics_reports_within_family_tactical_miss() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Two play-character actions share the same play slot. The model picks the wrong hand index,
    # so family and slot stay correct while the exact within-family choice is wrong.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 1.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, 19] = -4.0

    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
    )

    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_loss"] > 0.0
