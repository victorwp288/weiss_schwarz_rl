from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.loss_policy_anchor_stage import apply_impala_policy_anchor_stage
from weiss_rl.learners.vtrace import VTraceTargets

from .impala_test_support import (
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    TinyPolicyValueModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_apply_impala_policy_anchor_stage_preserves_inputs_loss_and_metrics() -> None:
    anchor_loss = torch.tensor(0.75, dtype=torch.float32)
    calls: list[dict[str, Any]] = []
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_model = object()
    reset_before_step = torch.tensor([[False], [True]], dtype=torch.bool)
    inputs = SimpleNamespace(
        obs=obs,
        loss_mask=loss_mask,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_model=forward_model,
        reset_before_step=reset_before_step,
    )
    batch: dict[str, bool] = {"policy_anchor_batch": True}

    def fake_policy_anchor_loss_and_metrics(
        source_batch: Any,
        *,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
        factorized_result: Any,
        forward_model: Any,
        reset_before_step: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        calls.append(
            {
                "batch": source_batch,
                "obs": obs,
                "loss_mask": loss_mask,
                "packed_legal": packed_legal,
                "factorized_result": factorized_result,
                "forward_model": forward_model,
                "reset_before_step": reset_before_step,
            }
        )
        return anchor_loss, {"policy_anchor_weighted_loss": float(anchor_loss)}

    learner = SimpleNamespace(_policy_anchor_loss_and_metrics=fake_policy_anchor_loss_and_metrics)
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss + anchor_loss)
    assert result.policy_anchor_loss is anchor_loss
    assert result.policy_anchor_metrics["policy_anchor_weighted_loss"] == pytest.approx(0.75)
    assert calls == [
        {
            "batch": batch,
            "obs": obs,
            "loss_mask": loss_mask,
            "packed_legal": packed_legal,
            "factorized_result": factorized_result,
            "forward_model": forward_model,
            "reset_before_step": reset_before_step,
        }
    ]


def test_apply_impala_policy_anchor_stage_preserves_total_loss_when_anchor_disabled() -> None:
    learner = SimpleNamespace(
        _policy_anchor_loss_and_metrics=lambda *args, **kwargs: (
            None,
            {"policy_anchor_disabled": 1.0},
        )
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        packed_legal=None,
        factorized_result=None,
        forward_model=object(),
        reset_before_step=None,
    )
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch={},
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss)
    assert result.policy_anchor_loss is None
    assert result.policy_anchor_metrics == {"policy_anchor_disabled": 1.0}


def test_impala_learner_factorized_policy_anchor_penalizes_post_anchor_drift() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(
        model=model,
        policy_anchor_coef=0.5,
        policy_anchor_temperature=1.0,
    )
    learner._ensure_policy_anchor_model()
    with torch.no_grad():
        model.bias.fill_(2.0)
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
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["policy_anchor_coef_active"] == pytest.approx(0.5)
    assert metrics["policy_anchor_loss"] > 0.0
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(metrics["policy_anchor_loss"] * 0.5)
    assert metrics["policy_anchor_candidate_count"] == pytest.approx(float(packed_ids.shape[0]))
    assert model.factorized_candidate_logp_calls == 1
    assert learner._policy_anchor_model is not None


def test_impala_learner_reset_policy_anchor_refreshes_current_weights() -> None:
    model = TinyPolicyValueModel()
    learner = ImpalaLearner(model=model, policy_anchor_coef=0.5)
    learner._ensure_policy_anchor_model()

    with torch.no_grad():
        model.policy.bias.fill_(3.0)
    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is not None
    anchor_bias = dict(learner._policy_anchor_model.state_dict())["policy.bias"]
    assert torch.equal(anchor_bias, model.policy.bias.detach())


def test_impala_learner_reset_policy_anchor_clears_disabled_anchor() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    learner._ensure_policy_anchor_model()

    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is None


def test_impala_learner_factorized_margin_aux_uses_factorized_candidate_log_probs_without_public_teacher() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_margin_coef=1.0,
        teacher_action_margin=0.5,
        teacher_same_family_action_margin_coef=1.0,
        teacher_same_family_action_margin=0.5,
        teacher_public_heuristic_coef=0.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 0
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_action_margin_mean"] > 0.5
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_margin_mean"] > 0.5
