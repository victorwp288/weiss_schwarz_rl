from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from weiss_rl.envs import decision_env as decision_env_module
from weiss_rl.envs.decision_env import DecisionBoundaryEnv, EngineStatusCounters

from tests.weiss_rl.decision_env_test_support import FakePool


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
