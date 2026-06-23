from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv, EngineStatusCounters

from tests.weiss_rl.decision_env_test_support import FakePool, FakeWeissSim, mask_step


def test_hard_fail_raises_and_counts_fault_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool()
    reset_step = mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = mask_step(engine_status=np.array([7], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, engine_status_policy="hard_fail", counters=counters)
    env.reset()

    with pytest.raises(RuntimeError, match=r"fault_rows=1"):
        env.step(np.array([10], dtype=np.int64))

    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert len(fake_weiss_sim.rl.step_calls) == 1


def test_passthrough_preserves_faulted_step_without_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool()
    reset_step = mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = mask_step(engine_status=np.array([7], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, engine_status_policy="passthrough", counters=counters)
    env.reset()
    returned = env.step(np.array([10], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert pool.calls == []
    npt.assert_array_equal(returned.engine_status, np.array([7], dtype=np.uint8))
