from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from weiss_rl.core.action_catalog import ActionCatalog

from .impala_test_support import (
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def _family_index_by_name(action_catalog: ActionCatalog) -> dict[str, int]:
    return {family.name: index for index, family in enumerate(action_catalog.families)}


def _two_row_factorized_teacher_batch(
    action_catalog: ActionCatalog,
    *,
    include_teacher_action: bool,
) -> dict[str, Any]:
    family_index = _family_index_by_name(action_catalog)
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": np.asarray([0, 3, 7], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }
    if include_teacher_action:
        batch["teacher_action"] = np.asarray([[0], [11]], dtype=np.int64)
    return batch


def _one_row_move_source_batch(action_catalog: ActionCatalog) -> dict[str, Any]:
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_index = _family_index_by_name(action_catalog)
    packed_ids = np.asarray([move_action, action_catalog.pass_action_id], dtype=np.uint32)
    return {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": np.asarray([0, 2], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_move"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[int(move_decoded.to_slot or 0)]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[move_action]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }


def test_impala_learner_auxiliary_update_uses_factorized_same_family_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_same_family_action_coef=1.0,
    )

    metrics = learner.auxiliary_update(_two_row_factorized_teacher_batch(action_catalog, include_teacher_action=True))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_hand_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_hand_coef=1.0,
    )

    metrics = learner.auxiliary_update(_two_row_factorized_teacher_batch(action_catalog, include_teacher_action=True))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_teacher_action_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_coef=1.0,
    )

    metrics = learner.auxiliary_update(_two_row_factorized_teacher_batch(action_catalog, include_teacher_action=True))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_move_source_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_move_source_coef=1.0,
    )

    metrics = learner.auxiliary_update(_one_row_move_source_batch(action_catalog))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
    )

    metrics = learner.auxiliary_update(_two_row_factorized_teacher_batch(action_catalog, include_teacher_action=False))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
