from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.action_logp import packed_scores_action_logp_and_entropy

from .impala_test_support import (
    ImpalaLearner,
    TrunkStructuredTeacherModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_impala_learner_packed_raw_vtrace_rho_is_one_when_behavior_matches_policy() -> None:
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
    obs = np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32)
    actions = np.asarray([[5], [11]], dtype=np.int64)
    to_play_seat = np.asarray([[0], [1]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            legal_actions=legal_actions,
        )
        assert forward.packed_logits is not None
        behavior_logp, _entropy = packed_scores_action_logp_and_entropy(
            forward.packed_logits,
            torch.as_tensor(packed_ids, dtype=torch.long),
            torch.as_tensor(packed_offsets, dtype=torch.long),
            torch.from_numpy(actions),
            pass_action_id=action_catalog.pass_action_id,
        )
    learner._active_timing_metrics = {}
    batch = {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    torch.testing.assert_close(context["action_logp"], behavior_logp)
    torch.testing.assert_close(context["vtrace_rhos"], torch.ones_like(context["vtrace_rhos"]))
    assert metrics["target_behavior_logp_delta_abs_p99"] == pytest.approx(0.0)
    assert metrics["target_behavior_train_logp_delta_abs_p99"] == pytest.approx(0.0)
