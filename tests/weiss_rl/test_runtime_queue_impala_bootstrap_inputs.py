from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime

from .runtime_test_support import _make_runtime_unroll


def test_build_learner_batch_preserves_bootstrap_inputs_for_learner_values() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 3
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.zeros((2, 1), dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 3), dtype=np.bool_)),
        bootstrap_obs=np.array([[3.0]], dtype=np.float32),
        bootstrap_actor=np.array([1], dtype=np.int64),
        final_hidden_state=np.array([[[1.0, 2.0]]], dtype=np.float32),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=0.0,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert np.array_equal(batch["bootstrap_obs"], unroll.bootstrap_obs)
    assert np.array_equal(batch["bootstrap_actor"], unroll.bootstrap_actor)
    assert np.array_equal(batch["final_hidden_state"], unroll.final_hidden_state)
