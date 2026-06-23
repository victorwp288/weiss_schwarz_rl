from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.learners.action_logp import masked_action_logp_and_entropy

from .impala_test_support import ImpalaLearner, TinyPolicyValueModel, _simple_training_batch


def test_impala_learner_raw_vtrace_inputs_use_current_policy_logp_for_importance_weights() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": (action_logp - 2.0).cpu().numpy().astype(np.float32),
        "behavior_values": values.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, metrics = learner._loss_and_metrics(raw_batch)

    assert metrics["vtrace_rho_p50"] > 7.0
    assert metrics["vtrace_rho_p95"] > 7.0
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(1.0)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)
