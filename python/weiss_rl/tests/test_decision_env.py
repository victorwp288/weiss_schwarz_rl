from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from weiss_rl.envs.decision_env import DecisionBoundaryEnv, EngineStatusCounters


class FakePool:
    def __init__(self, reported_rows: int | None = None) -> None:
        self.reported_rows = reported_rows
        self.calls: list[tuple[np.ndarray, object]] = []

    def auto_reset_on_error_codes_into(self, codes: np.ndarray, out: object) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append((copied_codes, out))
        out.reset_applied = True
        return self.reported_rows


class FakeSim:
    def __init__(self, step_result: object, *, pool: object | None = None) -> None:
        self.step_result = step_result
        self.pool = pool
        self.reset_seed: int | None = None
        self.actions_seen: list[np.ndarray] = []

    def reset(self, seed: int | None = None) -> object:
        self.reset_seed = seed
        return SimpleNamespace(seed=seed)

    def step(self, actions: np.ndarray) -> object:
        self.actions_seen.append(np.asarray(actions).copy())
        return self.step_result


def test_hard_fail_raises_and_counts_fault_rows() -> None:
    counters = EngineStatusCounters()
    batch = SimpleNamespace(engine_status=np.array([0, 7, 2], dtype=np.int32))
    env = DecisionBoundaryEnv(FakeSim(batch), engine_status_policy="hard_fail", counters=counters)

    with pytest.raises(RuntimeError, match=r"fault_rows=2"):
        env.step(np.array([10, 11, 12], dtype=np.int64))

    assert counters.fault_rows == 2
    assert counters.best_effort_reset_rows == 0


def test_best_effort_reset_uses_pool_and_counts_reported_rows() -> None:
    counters = EngineStatusCounters()
    batch = SimpleNamespace(engine_status=np.array([0, 3, 0, 5], dtype=np.int16))
    pool = FakePool(reported_rows=2)
    env = DecisionBoundaryEnv(FakeSim(batch, pool=pool), counters=counters)

    returned = env.step(np.array([1, 2, 3, 4], dtype=np.int64))

    assert returned is batch
    assert getattr(batch, "reset_applied", False) is True
    assert counters.fault_rows == 2
    assert counters.best_effort_reset_rows == 2
    assert len(pool.calls) == 1
    npt.assert_array_equal(pool.calls[0][0], np.array([0, 3, 0, 5], dtype=np.int32))
    assert pool.calls[0][1] is batch


def test_best_effort_reset_is_a_noop_when_pool_support_is_missing() -> None:
    counters = EngineStatusCounters()
    batch = SimpleNamespace(engine_status=np.array([4], dtype=np.int8))
    env = DecisionBoundaryEnv(FakeSim(batch, pool=SimpleNamespace()), counters=counters)

    returned = env.step(np.array([9], dtype=np.int64))

    assert returned is batch
    assert not hasattr(batch, "reset_applied")
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0


def test_reset_forwards_seed_to_underlying_sim() -> None:
    sim = FakeSim(SimpleNamespace(engine_status=np.array([0], dtype=np.int8)))
    env = DecisionBoundaryEnv(sim)

    env.reset(seed=123)

    assert sim.reset_seed == 123
