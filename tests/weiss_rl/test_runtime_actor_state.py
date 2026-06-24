from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.artifacts.reproducibility import derive_actor_seed
from weiss_rl.runtime.components.actors.actor_state import (
    MIRROR_OPPONENT_POLICY_ID,
    _ActorState,
    actor_seed,
    build_actor_state,
)


@dataclass(frozen=True)
class _FakeBatch:
    ids_offsets: Any | None = None
    legal_action_meta: Any | None = None


@dataclass
class _FakeActorState:
    actor_id: int
    env: Any
    model: Any
    compiled_model: Any
    rng: np.random.Generator
    seat_hidden: torch.Tensor
    current_batch: _FakeBatch
    layout_name: str
    focal_seat_by_env: np.ndarray
    opponent_policy_id_by_env: np.ndarray
    opponent_hidden: torch.Tensor
    diverse_opponent_lane: bool
    force_model_policy_lane: bool
    fixed_opponent_policy_id_by_env: np.ndarray | None


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def initial_seat_hidden(self, envs_per_actor: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros((envs_per_actor, 2), device=device)


class _FakeEnv:
    def __init__(self) -> None:
        self.reset_seed: int | None = None

    def reset(self, *, seed: int) -> _FakeBatch:
        self.reset_seed = int(seed)
        return _FakeBatch()


def test_actor_seed_matches_runtime_contract() -> None:
    assert actor_seed(17, 3) == derive_actor_seed(17, actor_id=3)
    assert actor_seed(20260514, 0) == derive_actor_seed(20260514, actor_id=0)


def test_actor_state_container_preserves_runtime_defaults_and_slots() -> None:
    model = _FakeModel()
    batch = _FakeBatch()

    state = _ActorState(
        actor_id=1,
        env=_FakeEnv(),  # type: ignore[arg-type]
        model=model,
        compiled_model=None,
        rng=np.random.default_rng(123),
        seat_hidden=torch.zeros((2, 3)),
        current_batch=batch,  # type: ignore[arg-type]
        layout_name="mask",
        focal_seat_by_env=np.zeros((2,), dtype=np.int64),
        opponent_policy_id_by_env=np.full((2,), MIRROR_OPPONENT_POLICY_ID, dtype=object),
        opponent_hidden=torch.zeros((2, 3)),
        diverse_opponent_lane=False,
        force_model_policy_lane=True,
    )

    assert state.snapshot_version == 0
    assert state.next_unroll_seq == 0
    assert state.fixed_opponent_policy_id_by_env is None
    assert cast(Any, state).current_batch is batch
    assert getattr(state, "__dict__", None) is None


def test_build_actor_state_initializes_model_env_and_episode_roles() -> None:
    env = _FakeEnv()
    fixed_slots = np.array(["seeded"], dtype=object)
    assigned: list[tuple[int, np.ndarray]] = []

    def compile_model(model: _FakeModel) -> str:
        assert model.eval_called
        return "compiled"

    state = build_actor_state(
        actor_state_cls=_FakeActorState,
        model=_FakeModel(),
        actor_id=2,
        env=env,  # type: ignore[arg-type]
        layout_name="mask",
        base_seed=11,
        envs_per_actor=3,
        device=torch.device("cpu"),
        shared_actor_model=None,
        shared_compiled_actor_model=None,
        maybe_compile_actor_model=compile_model,
        legal_action_meta_from_ids=lambda _ids: None,
        fixed_opponent_policy_slots=lambda: fixed_slots,
        diverse_opponent_actor_count=3,
        diverse_model_actor_count=1,
        assign_episode_roles=lambda actor, done: assigned.append((actor.actor_id, done.copy())),
    )

    assert env.reset_seed == actor_seed(11, 2)
    assert state.actor_id == 2
    assert state.compiled_model == "compiled"
    assert state.model.eval_called
    assert state.seat_hidden.shape == (3, 2)
    assert state.opponent_hidden.shape == (3, 2)
    assert state.diverse_opponent_lane is True
    assert state.force_model_policy_lane is False
    assert state.fixed_opponent_policy_id_by_env is fixed_slots
    assert state.opponent_policy_id_by_env.tolist() == [MIRROR_OPPONENT_POLICY_ID] * 3
    assert len(assigned) == 1
    assert assigned[0][0] == 2
    np.testing.assert_array_equal(assigned[0][1], np.ones((3,), dtype=np.bool_))
