from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.auxiliary.teacher_auxiliary_request import (
    compute_impala_teacher_auxiliary,
    resolve_impala_teacher_auxiliary_coefficients,
    resolve_impala_teacher_auxiliary_factorized_inputs,
    resolve_impala_teacher_auxiliary_inputs,
    resolve_impala_teacher_auxiliary_labels,
    resolve_impala_teacher_auxiliary_packed_inputs,
)

from .impala_test_support import (
    ImpalaLearner,
    TinyStructuredTeacherModel,
    _teacher_aux_catalog,
)


def test_compute_impala_teacher_auxiliary_request_preserves_dense_teacher_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
        profile_timers=True,
    )
    cast(Any, learner)._active_timing_metrics = {}
    expected_shape = torch.Size((1, 1))
    result = compute_impala_teacher_auxiliary(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
        legal_mask=torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool),
        loss_mask=torch.ones(expected_shape, dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=expected_shape,
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        batch_value=lambda batch, key: batch.get(key),
    )

    assert result.loss > 0.0
    assert result.metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.metrics["teacher_family_loss"] > 0.0
    assert result.metrics["teacher_action_loss"] > 0.0
    assert "teacher_family_log_probs" in result.context
    assert cast(Any, learner)._active_timing_metrics["timer_learner_teacher_aux_ms"] >= 0.0


def test_resolve_impala_teacher_auxiliary_labels_preserves_time_major_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    expected_shape = torch.Size((1, 2))
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"], family_index["attack"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0, 1]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1, 0]], dtype=np.int64),
        "teacher_action": np.asarray([[0, 11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True, False]], dtype=np.bool_),
    }

    labels = resolve_impala_teacher_auxiliary_labels(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=expected_shape,
    )

    assert labels.family is not None
    assert labels.family.dtype == torch.long
    assert labels.family.shape == expected_shape
    assert labels.family.tolist() == [[family_index["main_play_character"], family_index["attack"]]]
    assert labels.slot is not None
    assert labels.slot.tolist() == [[0, 1]]
    assert labels.move_source is None
    assert labels.attack_type is not None
    assert labels.attack_type.tolist() == [[-1, 0]]
    assert labels.action is not None
    assert labels.action.tolist() == [[0, 11]]
    assert labels.valid is not None
    assert labels.valid.dtype == torch.bool
    assert labels.valid.tolist() == [[True, False]]


def test_resolve_impala_teacher_auxiliary_coefficients_names_all_teacher_knobs() -> None:
    learner = ImpalaLearner(
        teacher_family_coef=0.1,
        teacher_slot_coef=0.2,
        teacher_hand_coef=0.3,
        teacher_move_source_coef=0.4,
        teacher_attack_type_coef=0.5,
        teacher_action_coef=0.6,
        teacher_same_family_action_coef=0.7,
        teacher_action_margin_coef=0.8,
        teacher_action_margin=0.9,
        teacher_same_family_action_margin_coef=1.1,
        teacher_same_family_action_margin=1.2,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=1.3,
        teacher_public_heuristic_temperature=1.4,
        teacher_public_nonpass_over_pass_coef=1.5,
        teacher_public_nonpass_over_pass_margin=1.6,
        teacher_public_heuristic_families=("main_play_character",),
    )

    coefficients = resolve_impala_teacher_auxiliary_coefficients(learner)

    assert coefficients.family == pytest.approx(0.1)
    assert coefficients.slot == pytest.approx(0.2)
    assert coefficients.hand == pytest.approx(0.3)
    assert coefficients.move_source == pytest.approx(0.4)
    assert coefficients.attack_type == pytest.approx(0.5)
    assert coefficients.action == pytest.approx(0.6)
    assert coefficients.same_family_action == pytest.approx(0.7)
    assert coefficients.action_margin == pytest.approx(0.8)
    assert coefficients.action_margin_value == pytest.approx(0.9)
    assert coefficients.same_family_action_margin == pytest.approx(1.1)
    assert coefficients.same_family_action_margin_value == pytest.approx(1.2)
    assert coefficients.exact_action_families == ("attack",)
    assert coefficients.public_heuristic == pytest.approx(1.3)
    assert coefficients.public_heuristic_temperature == pytest.approx(1.4)
    assert coefficients.public_nonpass_over_pass == pytest.approx(1.5)
    assert coefficients.public_nonpass_over_pass_margin == pytest.approx(1.6)
    assert coefficients.public_heuristic_families == ("main_play_character",)


def test_resolve_impala_teacher_auxiliary_factorized_inputs_preserves_required_and_optional_fields() -> None:
    required = {
        "family_log_probs": torch.zeros((1, 1, 2)),
        "play_slot_log_probs": torch.ones((1, 1, 3)),
        "move_slot_log_probs": torch.full((1, 1, 4), 2.0),
        "attack_slot_log_probs": torch.full((1, 1, 5), 3.0),
        "attack_type_log_probs": torch.full((1, 1, 6), 4.0),
    }
    result = resolve_impala_teacher_auxiliary_factorized_inputs(SimpleNamespace(**required))

    assert result.family_log_probs is required["family_log_probs"]
    assert result.play_slot_log_probs is required["play_slot_log_probs"]
    assert result.move_source_log_probs is None
    assert result.move_slot_log_probs is required["move_slot_log_probs"]
    assert result.attack_slot_log_probs is required["attack_slot_log_probs"]
    assert result.attack_type_log_probs is required["attack_type_log_probs"]
    assert result.top_action_ids is None
    assert result.same_family_action_logp is None
    assert result.same_family_top_action_ids is None


def test_resolve_impala_teacher_auxiliary_packed_inputs_preserves_tuple_contract() -> None:
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()

    packed = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
    )
    dense = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=None,
        packed_view=packed_view,
    )

    assert packed.ids is ids
    assert packed.offsets is offsets
    assert packed.meta is meta
    assert packed.view is packed_view
    assert dense.ids is None
    assert dense.offsets is None
    assert dense.meta is None
    assert dense.view is packed_view


def test_resolve_impala_teacher_auxiliary_inputs_preserves_aggregate_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.25,
        teacher_public_heuristic_coef=0.75,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        same_family_action_logp=torch.tensor([[-0.5]]),
    )
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }

    result = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )

    assert result.labels.family is not None
    assert result.labels.family.tolist() == [[family_index["main_play_character"]]]
    assert result.labels.move_source is None
    assert result.coefficients.family == pytest.approx(0.25)
    assert result.coefficients.public_heuristic == pytest.approx(0.75)
    assert result.coefficients.public_heuristic_families == ("main_play_character",)
    assert result.packed.ids is ids
    assert result.packed.offsets is offsets
    assert result.packed.meta is meta
    assert result.packed.view is packed_view
    assert result.factorized.family_log_probs is factorized_result.family_log_probs
    assert result.factorized.play_slot_log_probs is factorized_result.play_slot_log_probs
    assert result.factorized.same_family_action_logp is factorized_result.same_family_action_logp
    assert result.factorized.same_family_top_action_ids is None
