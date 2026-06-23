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


def test_compute_structured_teacher_auxiliary_metrics_supervises_slot_groups_not_hand_indices() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Row 0: two different hand indices map to the same play slot. Slot supervision should
    # treat their combined probability mass as correct.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1: attack family with the correct attack type.
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        teacher_slot=torch.tensor([[0], [0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [11]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.1,
        attack_type_coef=0.05,
        action_coef=0.15,
        same_family_action_coef=0.2,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_loss"] < 0.05
    assert metrics["teacher_action_loss"] < 0.35
    assert metrics["teacher_same_family_action_loss"] < 0.35


def test_compute_structured_teacher_auxiliary_metrics_groups_main_move_targets_by_destination_slot() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_actions_by_target: dict[int, list[int]] = {}
    for action_id in range(action_catalog.action_space_size):
        decoded = action_catalog.decode(action_id)
        if decoded.family != "main_move" or decoded.to_slot is None:
            continue
        move_actions_by_target.setdefault(int(decoded.to_slot), []).append(int(action_id))
    target_slot, target_actions = next(
        (slot, action_ids) for slot, action_ids in move_actions_by_target.items() if len(action_ids) >= 2
    )
    preferred_move, alternate_move = target_actions[:2]

    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [preferred_move, alternate_move, action_catalog.pass_action_id]] = True
    logits[0, 0, preferred_move] = 1.0
    logits[0, 0, alternate_move] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -4.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        "teacher_slot": torch.tensor([[target_slot]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[preferred_move]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 1.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 1.0,
    }

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)
    assert dense_metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert dense_metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert dense_metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_matches_packed_meta_path() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0
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

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)
