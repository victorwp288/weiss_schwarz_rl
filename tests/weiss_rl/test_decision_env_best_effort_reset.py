from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv, EngineStatusCounters

from tests.weiss_rl.decision_env_test_support import (
    FakeMaskOut,
    FakeNoMaskOut,
    FakePool,
    FakeWeissSim,
    ids_step,
    mask_step,
)


def test_best_effort_reset_mask_uses_pool_contract_and_counts_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    mask_reset = mask_step(engine_status=np.array([0], dtype=np.uint8))
    pool = FakePool(reported_rows=1, mask_reset_result=mask_reset)
    reset_step = mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = mask_step(engine_status=np.array([3], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([1], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    assert len(pool.calls) == 1
    method_name, reset_codes, reset_out = pool.calls[0]
    assert method_name == "mask"
    assert isinstance(reset_out, FakeMaskOut)
    assert getattr(reset_out, "reset_applied", False) is True
    assert reset_codes.dtype == np.uint8
    assert reset_codes.flags.c_contiguous
    npt.assert_array_equal(reset_codes, np.array([3], dtype=np.uint8))
    npt.assert_array_equal(returned.engine_status, np.array([0], dtype=np.uint8))


def test_best_effort_reset_ids_offsets_uses_nomask_contract_and_refills_legal_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = EngineStatusCounters()
    reset_step = ids_step(engine_status=np.array([0], dtype=np.uint8))
    faulted_step = ids_step(engine_status=np.array([5], dtype=np.uint8))
    nomask_reset = SimpleNamespace(
        obs=np.array([[7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
        rewards=np.array([0.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([1], dtype=np.int32),
        decision_kind=np.array([0], dtype=np.int32),
        decision_id=np.array([21], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        spec_hash=np.array([11], dtype=np.uint64),
    )
    pool = FakePool(
        reported_rows=1,
        nomask_reset_result=nomask_reset,
        legal_ids_after_reset=np.array([2, 4], dtype=np.uint32),
        legal_action_meta_after_reset=np.array([[6, 0, 0, 0], [6, 1, 0, 0]], dtype=np.uint16),
        legal_offsets_after_reset=np.array([0, 2], dtype=np.int32),
    )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=faulted_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality="ids_offsets", counters=counters)
    env.reset()
    returned = env.step(np.array([1], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    assert len(pool.calls) == 1
    method_name, reset_codes, reset_out = pool.calls[0]
    assert method_name == "nomask"
    assert isinstance(reset_out, FakeNoMaskOut)
    assert getattr(reset_out, "reset_applied", False) is True
    npt.assert_array_equal(reset_codes, np.array([5], dtype=np.uint8))
    assert len(pool.legal_refill_calls) == 1
    assert returned.ids_offsets is not None
    legal_ids, legal_offsets = returned.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([2, 4], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2], dtype=np.int32))
    assert returned.legal_action_meta is not None
    npt.assert_array_equal(returned.legal_action_meta, np.array([[6, 0, 0, 0], [6, 1, 0, 0]], dtype=np.uint16))
    npt.assert_array_equal(returned.actor, np.array([1], dtype=np.int32))
    npt.assert_array_equal(returned.engine_status, np.array([0], dtype=np.uint8))
    npt.assert_array_equal(returned.obs, np.array([[7, 8, 9, 10]], dtype=np.int16))


def test_best_effort_reset_ids_offsets_only_replaces_fault_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    reset_step = SimpleNamespace(
        obs=np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int16),
        rewards=np.array([0.0, 0.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([0, 1], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([10, 20], dtype=np.int64),
        engine_status=np.array([0, 0], dtype=np.uint8),
        spec_hash=np.array([11, 22], dtype=np.uint64),
        legal_ids=np.array([1, 5, 7, 8], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.int32),
    )
    faulted_step = SimpleNamespace(
        obs=np.array([[3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.int16),
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        actor=np.array([3, 4], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([30, 40], dtype=np.int64),
        engine_status=np.array([0, 9], dtype=np.uint8),
        spec_hash=np.array([33, 44], dtype=np.uint64),
        legal_ids=np.array([1, 5, 7, 8], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4], dtype=np.int32),
    )
    nomask_reset = SimpleNamespace(
        obs=np.array([[99.0, 99.0, 99.0, 99.0], [8.0, 8.0, 8.0, 8.0]], dtype=np.float32),
        rewards=np.array([9.0, 0.0], dtype=np.float32),
        terminated=np.array([True, False]),
        truncated=np.array([True, False]),
        actor=np.array([9, 6], dtype=np.int32),
        decision_kind=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([99, 60], dtype=np.int64),
        engine_status=np.array([7, 0], dtype=np.uint8),
        spec_hash=np.array([99, 66], dtype=np.uint64),
    )
    pool = FakePool(
        reported_rows=1,
        envs_len=2,
        nomask_reset_result=nomask_reset,
        legal_ids_after_reset=np.array([9, 10, 11, 12], dtype=np.uint32),
        legal_offsets_after_reset=np.array([0, 2, 4], dtype=np.int32),
    )
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=faulted_step)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, legality="ids_offsets", counters=counters)
    env.reset()
    returned = env.step(np.array([1, 7], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    npt.assert_array_equal(returned.obs, np.array([[3, 3, 3, 3], [8, 8, 8, 8]], dtype=np.int16))
    npt.assert_array_equal(returned.reward, np.array([1.0, 0.0], dtype=np.float32))
    npt.assert_array_equal(returned.actor, np.array([3, 6], dtype=np.int32))
    npt.assert_array_equal(returned.decision_id, np.array([30, 60], dtype=np.int64))
    npt.assert_array_equal(returned.engine_status, np.array([0, 0], dtype=np.uint8))
    assert returned.ids_offsets is not None
    legal_ids, legal_offsets = returned.ids_offsets
    npt.assert_array_equal(legal_ids, np.array([1, 5, 11, 12], dtype=np.uint32))
    npt.assert_array_equal(legal_offsets, np.array([0, 2, 4], dtype=np.int32))


def test_best_effort_reset_is_a_noop_when_pool_support_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = SimpleNamespace(envs_len=1, action_space=52)
    reset_step = mask_step(engine_status=np.array([0], dtype=np.uint8))
    step_result = mask_step(engine_status=np.array([4], dtype=np.uint8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([9], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
