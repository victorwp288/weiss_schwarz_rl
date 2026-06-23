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


def _one_row_public_heuristic_batch(action_catalog: ActionCatalog) -> dict[str, Any]:
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    return {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": np.asarray([0, 3], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }


def test_impala_learner_auxiliary_update_uses_factorized_public_heuristic_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
    )

    metrics = learner.auxiliary_update(_one_row_public_heuristic_batch(action_catalog))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.trunk_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_loss"] > 0.0
    assert metrics["teacher_public_heuristic_top1_mass"] < 0.1


def test_impala_learner_auxiliary_update_averages_multiple_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
    )

    metrics = learner.auxiliary_update(_one_row_public_heuristic_batch(action_catalog))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 3
    assert model.public_target_profiles == ["base", "aggressive", "control"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_auxiliary_update_cycles_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
    )
    learner.update_count = 1

    metrics = learner.auxiliary_update(_one_row_public_heuristic_batch(action_catalog))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["aggressive"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_public_heuristic_profiles_fall_back_to_base_after_end_update() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=0,
    )
    learner.update_count = 1

    learner.auxiliary_update(_one_row_public_heuristic_batch(action_catalog))

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["base"]
