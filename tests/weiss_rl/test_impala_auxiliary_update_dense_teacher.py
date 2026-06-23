from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.core.action_catalog import ActionCatalog

from .impala_test_support import (
    ImpalaLearner,
    TinyStructuredTeacherModel,
    _teacher_aux_catalog,
)


def _family_index_by_name(action_catalog: ActionCatalog) -> dict[str, int]:
    return {family.name: index for index, family in enumerate(action_catalog.families)}


def test_impala_learner_auxiliary_update_optimizes_teacher_only_loss() -> None:
    torch.manual_seed(0)

    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((2, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    legal_mask[1, 0, [10, 11, 12, 19]] = 1
    family_index = _family_index_by_name(action_catalog)
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_auxiliary_update_handles_batches_without_valid_teacher_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((1, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    family_index = _family_index_by_name(action_catalog)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[False]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0
