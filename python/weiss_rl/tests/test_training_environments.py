from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from weiss_rl.training import environments


class _FakeDecisionBoundaryEnv:
    def __init__(self, pool: object, **kwargs: Any) -> None:
        self.pool = pool
        self.kwargs = kwargs


def _stack(*, max_no_progress: int | None = 7) -> SimpleNamespace:
    curriculum = None
    if max_no_progress is not None:
        curriculum = SimpleNamespace(simulator={"max_no_progress_decisions": max_no_progress})
    return SimpleNamespace(config=SimpleNamespace(curriculum=curriculum))


def test_spec_dimensions_reads_observation_and_action_contract() -> None:
    contract = SimpleNamespace(
        spec_bundle={
            "observation": {"obs_len": "17"},
            "action": {"action_space_size": "91"},
        }
    )

    assert environments.spec_dimensions(contract) == (17, 91)


def test_build_training_env_preserves_mask_layout_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        environments,
        "env_pool_config",
        lambda stack, *, seed: {"max_decisions": 11, "max_ticks": 22, "seed": seed},
    )

    def _fake_make_pool(config: dict[str, Any], *, profile: str, num_envs: int):
        observed["make_pool"] = {"config": config, "profile": profile, "num_envs": num_envs}
        return object(), "mask"

    monkeypatch.setattr(environments, "make_env_pool_from_config", _fake_make_pool)
    monkeypatch.setattr(environments, "DecisionBoundaryEnv", _FakeDecisionBoundaryEnv)

    env = cast(Any, environments.build_training_env(_stack(), profile="fast", num_envs=3, seed=123))

    assert observed["make_pool"] == {
        "config": {"max_decisions": 11, "max_ticks": 22, "seed": 123},
        "profile": "fast",
        "num_envs": 3,
    }
    assert env.kwargs == {
        "legality": "mask",
        "engine_status_policy": "hard_fail",
        "max_decisions": 11,
        "max_ticks": 22,
        "max_no_progress_decisions": 7,
    }


def test_build_ids_eval_env_preserves_pinned_ids_layout_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environments,
        "env_pool_config",
        lambda stack, *, seed: {"max_decisions": 5, "max_ticks": 9, "seed": seed},
    )
    monkeypatch.setattr(
        environments,
        "make_env_pool_from_config",
        lambda config, *, profile, num_envs: (object(), "i16_legal_ids"),
    )
    monkeypatch.setattr(environments, "DecisionBoundaryEnv", _FakeDecisionBoundaryEnv)

    env = cast(Any, environments.build_ids_eval_env(_stack(max_no_progress=None), seed=456, pass_action_id=0))

    assert env.kwargs == {
        "legality": "ids_offsets",
        "pass_action_id": 0,
        "engine_status_policy": "hard_fail",
        "max_decisions": 5,
        "max_ticks": 9,
        "max_no_progress_decisions": None,
    }


def test_environment_builders_reject_unexpected_legality_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environments, "env_pool_config", lambda stack, *, seed: {"max_decisions": 1, "max_ticks": 1})
    monkeypatch.setattr(
        environments,
        "make_env_pool_from_config",
        lambda config, *, profile, num_envs: (object(), "unexpected"),
    )

    with pytest.raises(RuntimeError, match="expects mask legality"):
        environments.build_training_env(_stack(), profile="fast", num_envs=1, seed=1)

    with pytest.raises(RuntimeError, match="requires ids-based legality"):
        environments.build_ids_eval_env(_stack(), seed=1, pass_action_id=0)
