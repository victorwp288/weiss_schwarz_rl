from __future__ import annotations

from dataclasses import replace

import numpy as np
from weiss_rl.runtime.components.batching import build_impala_learner_batch

from .runtime_test_support import _make_runtime_unroll


def test_build_impala_batch_concatenates_opponent_context_index() -> None:
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        opponent_context_index=np.asarray([[1]], dtype=np.int16),
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        opponent_context_index=np.asarray([[2]], dtype=np.int16),
    )

    batch = build_impala_learner_batch(
        [unroll_a, unroll_b],
        action_dim=1,
        gamma=1.0,
        truncation_reward=0.0,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["opponent_context_index"].dtype == np.int16
    assert batch["opponent_context_index"].tolist() == [[1, 2]]
