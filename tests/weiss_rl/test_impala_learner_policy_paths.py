from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.vtrace import VTraceTargets

from .impala_test_support import (
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    TinyPolicyValueModel,
    TrunkStructuredTeacherModel,
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
)


def test_impala_learner_packed_legal_actions_match_dense_mask_loss() -> None:
    torch.manual_seed(0)
    dense_model = TinyPolicyValueModel()
    packed_model = TinyPolicyValueModel()
    packed_model.load_state_dict(dense_model.state_dict())
    dense_learner = ImpalaLearner(model=dense_model, pass_action_id=2)
    packed_learner = ImpalaLearner(model=packed_model, pass_action_id=2)

    legal_mask = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    actions = np.asarray([[0], [2]], dtype=np.int64)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": actions,
        "legal_mask": legal_mask,
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask)
    packed_batch = dict(batch)
    packed_batch["legal_actions"] = LegalActionBatch.from_packed(packed_ids, packed_offsets)
    packed_batch["legal_mask"] = None

    dense_loss, dense_metrics = dense_learner._loss_and_metrics(batch)
    packed_loss, packed_metrics = packed_learner._loss_and_metrics(packed_batch)

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_batch["legal_mask"] is None
    assert dense_metrics == pytest.approx(packed_metrics)


def test_impala_learner_dense_trajectory_retention_is_separate_from_policy_train_mask() -> None:
    torch.manual_seed(0)
    base_model = TinyPolicyValueModel(action_dim=2)
    retention_model = TinyPolicyValueModel(action_dim=2)
    retention_model.load_state_dict(base_model.state_dict())
    base_learner = ImpalaLearner(model=base_model)
    retention_learner = ImpalaLearner(model=retention_model, trajectory_retention_coef=0.4)
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    base_loss, _base_metrics = base_learner._loss_and_metrics(batch)
    retention_loss, retention_metrics = retention_learner._loss_and_metrics(batch)

    assert retention_metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert retention_metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_supported_fraction"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_weighted_loss"] > 0.0
    assert float(retention_loss.detach()) == pytest.approx(
        float(base_loss.detach()) + retention_metrics["trajectory_retention_weighted_loss"]
    )


def test_impala_learner_uses_factorized_legal_policy_path_for_loss_and_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        profile_timers=True,
        trajectory_retention_coef=0.06,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.1], [0.2]], dtype=np.float32),
            pg_advantages=np.asarray([[1.0], [0.5]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    loss, metrics = learner._loss_and_metrics(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert float(loss.detach()) != 0.0
    assert model.factorized_calls == 1
    assert learner._active_timing_metrics["timer_learner_factorized_policy_ms"] >= 0.0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(7.0)
    assert metrics["entropy"] > 0.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(0.25)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(0.015)


def test_impala_learner_restricts_packed_policy_scoring_to_train_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[5], [11]], dtype=np.int64),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=action_catalog.action_space_size,
        ),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": np.asarray([[-2.0], [-3.0]], dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert float(loss.detach()) != 0.0
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.scorer_row_count == 1
    assert model.scorer_candidate_count == 3
    assert learner._active_timing_metrics["packed_candidate_train_rows"] == pytest.approx(1.0)
    assert learner._active_timing_metrics["packed_candidate_train_count"] == pytest.approx(3.0)
    assert float(context["vtrace_rhos"][1, 0]) == pytest.approx(1.0)
    assert float(context["vtrace_rhos"][0, 0]) > 1.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
