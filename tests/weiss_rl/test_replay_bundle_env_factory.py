from __future__ import annotations

from typing import Any

import pytest
from weiss_rl.replay import runner as replay_runner

from .replay_bundle_test_support import rerun_contract


def test_build_replay_env_uses_fast_pool_factory_for_default_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = rerun_contract()
    pool = object()
    pool_factory_calls: dict[str, Any] = {}
    wrapper_calls: dict[str, Any] = {}

    class FakeDecisionBoundaryEnv:
        def __init__(self, pool_arg: Any, **kwargs: Any) -> None:
            wrapper_calls["pool"] = pool_arg
            wrapper_calls.update(kwargs)

    def fake_make_env_pool_from_config(
        env_config: dict[str, Any],
        *,
        profile: str,
        num_envs: int | None = None,
    ) -> tuple[object, str]:
        pool_factory_calls["env_config"] = dict(env_config)
        pool_factory_calls["profile"] = profile
        pool_factory_calls["num_envs"] = num_envs
        return pool, "i16_legal_ids"

    monkeypatch.setattr(replay_runner, "make_env_pool_from_config", fake_make_env_pool_from_config)
    monkeypatch.setattr(replay_runner, "DecisionBoundaryEnv", FakeDecisionBoundaryEnv)

    env = replay_runner.build_replay_env(contract)

    assert isinstance(env, FakeDecisionBoundaryEnv)
    assert pool_factory_calls == {
        "env_config": {
            "max_decisions": contract.max_decisions,
            "max_ticks": contract.max_ticks,
            "observation_visibility": contract.observation_visibility,
            "seed": 0,
            "reward_json": contract.reward_json,
            "curriculum_json": contract.curriculum_json,
            "deck": contract.deck,
            "opponent_deck": contract.opponent_deck,
        },
        "profile": "fast",
        "num_envs": 1,
    }
    assert wrapper_calls["pool"] is pool
    assert wrapper_calls["legality"] == "ids_offsets"
    assert wrapper_calls["engine_status_policy"] == "passthrough"
    assert pool_factory_calls["env_config"]["reward_json"] == contract.reward_json
    assert pool_factory_calls["env_config"]["curriculum_json"] == contract.curriculum_json
