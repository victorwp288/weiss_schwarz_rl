from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.loss_metrics import chosen_action_outcome_metrics as chosen_action_outcome_metrics_impl
from weiss_rl.learners.vtrace import VTraceTargets

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel, _mulligan_metric_catalog


def test_impala_learner_reports_reward_advantage_and_chosen_action_metrics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), pass_action_id=1)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.25], [-0.5]], dtype=np.float32),
            pg_advantages=np.asarray([[1.5], [-0.25]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "rewards": np.asarray([[0.0], [1.0]], dtype=np.float32),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["reward_std"] == pytest.approx(0.5)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["reward_min"] == pytest.approx(0.0)
    assert metrics["reward_max"] == pytest.approx(1.0)
    assert metrics["reward_nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["reward_positive_fraction"] == pytest.approx(0.5)
    assert metrics["reward_negative_fraction"] == pytest.approx(0.0)
    assert metrics["advantage_mean"] == pytest.approx(0.625)
    assert metrics["advantage_abs_mean"] == pytest.approx(0.875)
    assert metrics["target_mean"] == pytest.approx(-0.125)
    assert metrics["target_abs_mean"] == pytest.approx(0.375)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.5)
    assert metrics["chosen_pass_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_pass_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_nonpass_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)


def test_impala_learner_reports_family_chosen_action_outcome_metrics() -> None:
    catalog = _mulligan_metric_catalog()

    metrics = chosen_action_outcome_metrics_impl(
        actions=torch.tensor([[0], [1], [3], [5], [8]], dtype=torch.long),
        loss_mask=torch.tensor([[True], [True], [False], [True], [True]]),
        rewards=torch.tensor([[0.0], [1.0], [99.0], [2.0], [3.0]], dtype=torch.float32),
        advantages=torch.tensor([[0.5], [-0.25], [99.0], [1.25], [-1.0]], dtype=torch.float32),
        action_catalog=catalog,
        pass_action_id=catalog.pass_action_id,
    )

    assert metrics["chosen_mulligan_confirm_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_confirm_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_mulligan_confirm_train_advantage_mean"] == pytest.approx(0.5)
    assert metrics["chosen_mulligan_select_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_select_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_mulligan_select_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_attack_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_attack_train_advantage_mean"] == pytest.approx(1.25)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_main_play_character_train_fraction"] == pytest.approx(0.0)
