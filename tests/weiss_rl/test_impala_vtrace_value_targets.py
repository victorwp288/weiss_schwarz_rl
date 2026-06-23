from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from weiss_rl.learners.action_logp import masked_action_logp_and_entropy

from .impala_test_support import ImpalaLearner, SeatAwareTinyPolicyValueModel, TinyPolicyValueModel


def test_impala_learner_raw_vtrace_inputs_use_current_learner_values_for_targets() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, -0.5]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)

    with torch.no_grad():
        forward = learner._forward_time_major(torch.from_numpy(obs))
        logits = forward.logits
        assert logits is not None
        values = forward.values
        action_logp, _entropy = masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )

    log_rho = -0.2
    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": (action_logp - log_rho).cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), 123.0, dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 2.4,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    expected_rho = float(np.exp(log_rho))
    expected_targets = values.detach() * (1.0 - expected_rho)
    assert torch.allclose(context["targets"], expected_targets, atol=1.0e-6)


def test_impala_learner_raw_vtrace_inputs_can_bootstrap_from_current_model() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=SeatAwareTinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)
    to_play_seat = np.asarray([[0]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    bootstrap_obs = np.asarray([[2.0, 0.0]], dtype=np.float32)
    bootstrap_actor = np.asarray([1], dtype=np.int64)
    final_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            to_play_seat=to_play_seat,
            initial_hidden_state=initial_hidden_state,
        )
        logits = forward.logits
        assert logits is not None
        action_logp, _entropy = masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )
        model = cast(Any, learner.model)
        expected_bootstrap = model.value_seat_aware(
            torch.from_numpy(bootstrap_obs),
            torch.from_numpy(bootstrap_actor),
            torch.from_numpy(final_hidden_state),
        )

    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": action_logp.cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), -77.0, dtype=np.float32),
        "bootstrap_value": np.full((1,), 123.0, dtype=np.float32),
        "bootstrap_obs": bootstrap_obs,
        "bootstrap_actor": bootstrap_actor,
        "final_hidden_state": final_hidden_state,
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    assert torch.allclose(context["targets"], expected_bootstrap.reshape(1, 1), atol=1.0e-6)
