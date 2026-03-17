from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import (
    DecisionBoundaryBatch,
    DecisionBoundaryEnv,
    EngineStatusCounters,
    LegalMode,
    _pack_batch,
    _validate_actions,
)

_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


class FakePool:
    def __init__(self, reported_rows: int | None = None, *, envs_len: int = 1, action_space: int = 52) -> None:
        self.reported_rows = reported_rows
        self.envs_len = envs_len
        self.action_space = action_space
        self.calls: list[tuple[np.ndarray, object]] = []

    def auto_reset_on_error_codes_into(self, codes: np.ndarray, out: Any) -> int | None:
        copied_codes = np.asarray(codes).copy()
        self.calls.append((copied_codes, out))
        out.reset_applied = True
        return self.reported_rows


class FakeRlApi:
    def __init__(self, *, reset_result: object, step_result: object) -> None:
        self.reset_result = reset_result
        self.step_result = step_result
        self.reset_calls: list[tuple[object, str]] = []
        self.step_calls: list[tuple[object, np.ndarray, str]] = []

    def reset_rl(self, pool: object, *, layout: str) -> object:
        self.reset_calls.append((pool, layout))
        return self.reset_result

    def step_rl(self, pool: object, actions: np.ndarray, *, layout: str) -> object:
        self.step_calls.append((pool, np.asarray(actions).copy(), layout))
        return self.step_result


class FakeWeissSim:
    PASS_ACTION_ID = 51

    def __init__(self, *, reset_result: object, step_result: object) -> None:
        self.rl = FakeRlApi(reset_result=reset_result, step_result=step_result)


def _load_weiss_sim():
    return pytest.importorskip("weiss_sim")


def _fixture_db_path() -> Path:
    weiss_sim = _load_weiss_sim()
    return Path(weiss_sim.__file__).resolve().parents[1] / "tests" / "fixtures" / "cards.wsdb"


def _make_env(*, legality: LegalMode, num_envs: int = 2, seed: int = 123) -> DecisionBoundaryEnv:
    return DecisionBoundaryEnv.create(
        legality=legality,
        mode="train",
        num_envs=num_envs,
        db_path=str(_fixture_db_path()),
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=seed,
    )


def _first_legal_from_mask(mask: np.ndarray, *, pass_action_id: int) -> np.ndarray:
    actions = np.empty((int(mask.shape[0]),), dtype=np.uint32)
    for env_index in range(mask.shape[0]):
        legal = np.flatnonzero(mask[env_index])
        actions[env_index] = pass_action_id if legal.size == 0 else int(legal[0])
    return actions


def _first_legal_from_ids(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    pass_action_id: int,
) -> np.ndarray:
    num_envs = int(legal_offsets.shape[0]) - 1
    actions = np.empty((num_envs,), dtype=np.uint32)
    for env_index in range(num_envs):
        start = int(legal_offsets[env_index])
        end = int(legal_offsets[env_index + 1])
        actions[env_index] = pass_action_id if start == end else int(legal_ids[start])
    return actions


def _illegal_action_from_mask(mask: np.ndarray) -> int:
    illegal = np.flatnonzero(~mask[0].astype(bool))
    if illegal.size == 0:
        raise AssertionError("expected at least one illegal action")
    return int(illegal[0])


def _illegal_action_from_ids(legal_ids: np.ndarray, legal_offsets: np.ndarray, *, action_space: int) -> int:
    start = int(legal_offsets[0])
    end = int(legal_offsets[1])
    legal = set(int(action) for action in legal_ids[start:end].tolist())
    for action in range(action_space):
        if action not in legal:
            return action
    raise AssertionError("expected at least one illegal action")


def _done_step(legality: LegalMode) -> SimpleNamespace:
    step_kwargs = {
        "obs": np.zeros((1, 4), dtype=np.float32),
        "rewards": np.zeros((1,), dtype=np.float32),
        "terminated": np.array([False]),
        "truncated": np.array([True]),
        "actor": np.array([-1], dtype=np.int32),
        "decision_id": np.array([17], dtype=np.int64),
        "engine_status": np.array([0], dtype=np.int32),
    }
    if legality == "mask":
        return SimpleNamespace(**step_kwargs, masks=np.zeros((1, 52), dtype=np.uint8))
    return SimpleNamespace(
        **step_kwargs,
        legal_ids=np.array([], dtype=np.uint32),
        legal_offsets=np.array([0, 0], dtype=np.int32),
    )


def _mask_step(*, engine_status: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        rewards=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([17], dtype=np.int64),
        engine_status=np.asarray(engine_status),
        masks=np.ones((1, 52), dtype=np.uint8),
    )


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_done_batch_with_actor_minus_one_and_empty_legality_is_pass_only(legality: LegalMode) -> None:
    pass_action_id = 51
    batch = _pack_batch(_done_step(legality), legality=legality)

    assert int(batch.actor[0]) == -1
    assert bool(batch.truncated[0])

    _validate_actions(np.array([pass_action_id], dtype=np.uint32), batch, pass_action_id=pass_action_id)
    with pytest.raises(ValueError, match=f"expected pass action {pass_action_id}"):
        _validate_actions(np.array([0], dtype=np.uint32), batch, pass_action_id=pass_action_id)


def test_decision_env_create_uses_lazy_weiss_sim_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("weiss_rl.envs.decision_env")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "weiss_sim":
            raise ImportError("missing simulator")
        return real_import_module(name, package)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="weiss_sim"):
        module.DecisionBoundaryEnv.create(legality="mask", mode="train", num_envs=1)


def test_mask_mode_reset_and_step_return_typed_batch() -> None:
    env = _make_env(legality="mask")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is not None
        assert batch.ids_offsets is None
        assert batch.obs.shape[0] == env.num_envs
        assert batch.reward.shape == (env.num_envs,)
        assert batch.terminated.shape == (env.num_envs,)
        assert batch.truncated.shape == (env.num_envs,)
        assert batch.decision_id.shape == (env.num_envs,)
        assert batch.engine_status.shape == (env.num_envs,)
        assert np.array_equal(batch.to_play, batch.actor)

        next_batch = env.step(_first_legal_from_mask(batch.mask, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is not None
        assert next_batch.ids_offsets is None
        assert next_batch.obs.shape[0] == env.num_envs
        assert next_batch.reward.shape == (env.num_envs,)
    finally:
        env.close()


def test_ids_offsets_mode_reset_and_step_return_typed_batch() -> None:
    env = _make_env(legality="ids_offsets")
    try:
        batch = env.reset()
        assert isinstance(batch, DecisionBoundaryBatch)
        assert batch.num_envs == env.num_envs
        assert batch.mask is None
        assert batch.ids_offsets is not None
        legal_ids, legal_offsets = batch.ids_offsets
        assert legal_ids.ndim == 1
        assert legal_offsets.shape == (env.num_envs + 1,)
        assert np.array_equal(batch.to_play, batch.actor)

        next_batch = env.step(_first_legal_from_ids(legal_ids, legal_offsets, pass_action_id=env.pass_action_id))
        assert isinstance(next_batch, DecisionBoundaryBatch)
        assert next_batch.mask is None
        assert next_batch.ids_offsets is not None
        assert next_batch.reward.shape == (env.num_envs,)
    finally:
        env.close()


@pytest.mark.parametrize("legality", ["mask", "ids_offsets"])
def test_step_rejects_illegal_action(legality: LegalMode) -> None:
    env = _make_env(legality=legality, num_envs=1)
    try:
        batch = env.reset()
        if batch.mask is not None:
            illegal_action = _illegal_action_from_mask(batch.mask)
        else:
            assert batch.ids_offsets is not None
            legal_ids, legal_offsets = batch.ids_offsets
            illegal_action = _illegal_action_from_ids(
                legal_ids,
                legal_offsets,
                action_space=env.action_space,
            )

        with pytest.raises(ValueError, match="illegal action"):
            env.step(np.array([illegal_action], dtype=np.uint32))
    finally:
        env.close()


def test_reset_rejects_explicit_seed() -> None:
    env = _make_env(legality="mask", num_envs=1)
    try:
        with pytest.raises(ValueError, match=r"reset\(seed=\.\.\.\)"):
            env.reset(seed=7)
    finally:
        env.close()


def test_hard_fail_raises_and_counts_fault_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool()
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.int32))
    step_result = _mask_step(engine_status=np.array([7], dtype=np.int32))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, engine_status_policy="hard_fail", counters=counters)
    env.reset()

    with pytest.raises(RuntimeError, match=r"fault_rows=1"):
        env.step(np.array([10], dtype=np.int64))

    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert len(fake_weiss_sim.rl.step_calls) == 1


def test_best_effort_reset_uses_pool_and_counts_reported_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = FakePool(reported_rows=1)
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.int32))
    step_result = _mask_step(engine_status=np.array([3], dtype=np.int16))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([1], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 1
    assert len(pool.calls) == 1
    npt.assert_array_equal(pool.calls[0][0], np.array([3], dtype=np.int32))
    assert pool.calls[0][1] is step_result
    assert getattr(step_result, "reset_applied", False) is True


def test_best_effort_reset_is_a_noop_when_pool_support_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = EngineStatusCounters()
    pool = SimpleNamespace(envs_len=1, action_space=52)
    reset_step = _mask_step(engine_status=np.array([0], dtype=np.int32))
    step_result = _mask_step(engine_status=np.array([4], dtype=np.int8))
    fake_weiss_sim = FakeWeissSim(reset_result=reset_step, step_result=step_result)
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    env = DecisionBoundaryEnv(pool, counters=counters)
    env.reset()
    returned = env.step(np.array([9], dtype=np.int64))

    assert isinstance(returned, DecisionBoundaryBatch)
    assert counters.fault_rows == 1
    assert counters.best_effort_reset_rows == 0
    assert not hasattr(step_result, "reset_applied")


def test_create_threads_engine_status_policy_and_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = FakePool(envs_len=2, action_space=52)
    fake_weiss_sim = SimpleNamespace(
        PASS_ACTION_ID=51,
        make_pool=lambda **kwargs: (pool, {"kwargs": kwargs}),
    )
    monkeypatch.setattr(decision_env_module, "_load_weiss_sim", lambda: fake_weiss_sim)

    counters = EngineStatusCounters()
    env = DecisionBoundaryEnv.create(
        legality="mask",
        engine_status_policy="hard_fail",
        counters=counters,
        mode="train",
        num_envs=2,
    )

    assert env.pool is pool
    assert env.pass_action_id == 51
    assert env.engine_status_policy == "hard_fail"
    assert env.counters is counters
