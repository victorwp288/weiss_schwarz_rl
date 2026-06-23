from __future__ import annotations

from typing import Any

import numpy as np
import torch
from weiss_rl.core.legal_actions import LegalActionBatch

from tests.weiss_rl.impala_test_support import (
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    TinyStructuredTeacherModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def make_paired_swing_dense_case() -> tuple[Any, ImpalaLearner, dict[str, Any]]:
    action_catalog = _teacher_aux_catalog()
    model = TinyStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = 0.0
        model.policy.bias[5] = 1.0
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "actions": np.asarray([[5]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=action_catalog.action_space_size,
        ),
    }
    return action_catalog, learner, batch


def make_factorized_paired_outcome_case() -> tuple[FactorizedStructuredTeacherModel, ImpalaLearner, dict[str, Any]]:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }
    return model, learner, batch


def make_factorized_paired_outcome_loss_case() -> tuple[
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    dict[str, Any],
]:
    model, learner, batch = make_factorized_paired_outcome_case()
    batch.update(
        {
            "preference_pair_id": np.asarray([[7], [7]], dtype=np.int64),
            "preference_role": np.asarray([[1], [0]], dtype=np.int64),
            "preference_group_id": np.asarray([[3], [3]], dtype=np.int64),
        }
    )
    return model, learner, batch
