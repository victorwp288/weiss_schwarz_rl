from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import DecisionBoundaryEnv, LegalMode

from tests.weiss_rl.decision_env_test_support import FakePool, FakeWeissSim


@pytest.mark.parametrize(
    ("legality", "expected_method"),
    [("mask", "mask"), ("ids_offsets", "ids_offsets")],
)
def test_reset_with_explicit_seed_uses_pool_seeded_reset_contract(
    legality: LegalMode,
    expected_method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool(envs_len=2)
    if legality == "mask":
        reset_step = SimpleNamespace(
            obs=np.zeros((2, 4), dtype=np.float32),
            rewards=np.zeros((2,), dtype=np.float32),
            terminated=np.zeros((2,), dtype=bool),
            truncated=np.zeros((2,), dtype=bool),
            actor=np.zeros((2,), dtype=np.int32),
            decision_kind=np.zeros((2,), dtype=np.int32),
            decision_id=np.zeros((2,), dtype=np.int64),
            engine_status=np.zeros((2,), dtype=np.uint8),
            spec_hash=np.zeros((2,), dtype=np.uint64),
            masks=np.zeros((2, 52), dtype=np.uint8),
        )
    else:
        reset_step = SimpleNamespace(
            obs=np.zeros((2, 4), dtype=np.int16),
            rewards=np.zeros((2,), dtype=np.float32),
            terminated=np.zeros((2,), dtype=bool),
            truncated=np.zeros((2,), dtype=bool),
            actor=np.zeros((2,), dtype=np.int32),
            decision_kind=np.zeros((2,), dtype=np.int32),
            decision_id=np.zeros((2,), dtype=np.int64),
            engine_status=np.zeros((2,), dtype=np.uint8),
            spec_hash=np.zeros((2,), dtype=np.uint64),
            legal_ids=np.array([], dtype=np.uint32),
            legal_offsets=np.array([0, 0, 0], dtype=np.int32),
        )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=reset_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality=legality)
    batch = env.reset(seed=77)

    assert fake_weiss_sim.rl.reset_calls == []
    assert len(pool.seeded_reset_calls) == 1
    method_name, indices, episode_seeds, out = pool.seeded_reset_calls[0]
    assert method_name == expected_method
    npt.assert_array_equal(indices, np.array([0, 1], dtype=np.int64))
    npt.assert_array_equal(episode_seeds, np.array([77, 77], dtype=np.uint64))
    npt.assert_array_equal(batch.episode_seed, np.array([77, 77], dtype=np.uint64))
    assert out is not None
