from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)

from .impala_test_support import (
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_compute_structured_teacher_auxiliary_metrics_supports_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)

    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
    }

    logits[0, 0, 0] = 4.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -5.0
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    logits[0, 0, 0] = 0.5
    logits[0, 0, 5] = 4.0
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    teacher_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    misaligned_view = _packed_structured_legal_view(
        logits=torch.tensor([4.0, 0.5, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=misaligned_view,
        **cast(Any, teacher_kwargs),
    )

    aligned_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=aligned_view,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_gates_public_heuristic_by_family() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    common_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "packed_view": packed_view,
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    allowed_loss, allowed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("main_play_character",),
        **cast(Any, common_kwargs),
    )
    gated_loss, gated_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("attack",),
        **cast(Any, common_kwargs),
    )

    assert float(allowed_loss.detach()) > 0.0
    assert allowed_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert allowed_metrics["teacher_public_heuristic_loss"] > 0.0
    assert float(gated_loss.detach()) == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_loss"] == pytest.approx(0.0)
