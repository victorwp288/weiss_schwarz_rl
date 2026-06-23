from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy

from .heuristic_public_test_support import _LEGAL_DECK


def test_simulator_native_heuristic_pool_matches_python_oracle_across_live_steps() -> None:
    weiss_sim = pytest.importorskip(
        "weiss_sim",
        reason="native heuristic pool parity test requires the optional simulator package",
    )

    env = DecisionBoundaryEnv.create(
        legality="ids_offsets",
        mode="train",
        num_envs=4,
        db_path=None,
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=321,
    )
    try:
        batch = env.reset()
        for _ in range(24):
            assert batch.ids_offsets is not None
            assert batch.legal_action_meta is not None
            legal_ids, legal_offsets = batch.ids_offsets
            policy = HeuristicPublicPolicy.from_spec_bundle(weiss_sim.spec_bundle())
            native_actions = np.zeros((env.num_envs,), dtype=np.uint16)
            env.pool.choose_heuristic_public_actions_into(
                np.arange(env.num_envs, dtype=np.uint32),
                native_actions,
            )
            oracle_actions = policy.choose_actions_from_meta_batch(
                np.asarray(batch.obs, dtype=np.int32),
                np.asarray(legal_ids, dtype=np.uint32),
                np.asarray(legal_offsets, dtype=np.uint32),
                np.asarray(batch.legal_action_meta, dtype=np.uint16),
            )
            npt.assert_array_equal(native_actions.astype(np.int64), oracle_actions)
            batch = env.step(native_actions.astype(np.int64))
    finally:
        env.close()
