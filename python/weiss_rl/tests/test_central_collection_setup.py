from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from weiss_rl.runtime_components import central_collection as central_collection_module
from weiss_rl.runtime_components.central_collection import QueueRuntimeCentralCollectionMixin
from weiss_rl.runtime_components.central_collection_setup import (
    actors_have_single_layout,
    build_central_actor_collection_setup,
    supports_structured_central_packed,
)


def _batch(*, dtype: Any = np.float32) -> SimpleNamespace:
    return SimpleNamespace(obs=np.zeros((3, 4), dtype=dtype))


def _actor(
    *,
    actor_id: int,
    layout_name: str = "i16_legal_ids",
    supports_legal_candidate_scoring: bool = True,
    dtype: Any = np.float32,
) -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=actor_id,
        layout_name=layout_name,
        current_batch=_batch(dtype=dtype),
        seat_hidden=torch.full((3, 2), float(actor_id + 1), dtype=torch.float32),
        env=SimpleNamespace(name=f"env-{actor_id}", max_decisions=10 + actor_id),
        supports_legal_candidate_scoring=supports_legal_candidate_scoring,
    )


def test_central_collection_setup_allocates_actor_state_timeouts_and_batches() -> None:
    actors = [_actor(actor_id=0, dtype=np.float64), _actor(actor_id=1, dtype=np.float64)]

    setup = build_central_actor_collection_setup(
        actors=actors,
        config=SimpleNamespace(unroll_length=2, envs_per_actor=3),
        observation_dim=4,
        trajectory_retention_enabled=True,
        actor_inference_model=lambda actor: actor,
        actor_timeout_limits=lambda env: {"max_decisions": int(env.max_decisions), "max_ticks": None},
    )

    assert setup.time_steps == 2
    assert setup.batch_size == 3
    assert setup.obs_dtype == np.dtype(np.float64)
    assert setup.batches == [actors[0].current_batch, actors[1].current_batch]
    assert setup.timeout_limits_by_actor == {
        0: {"max_decisions": 10, "max_ticks": None},
        1: {"max_decisions": 11, "max_ticks": None},
    }
    assert setup.structured_central_packed is True
    assert set(setup.states_by_actor) == {0, 1}
    state = setup.states_by_actor[0]
    assert state.obs.shape == (2, 3, 4)
    assert state.obs.dtype == np.float64
    assert state.trajectory_retention_valid is not None
    np.testing.assert_allclose(state.initial_hidden_state, np.ones((3, 2), dtype=np.float32))

    actors[0].seat_hidden.fill_(99.0)
    np.testing.assert_allclose(state.initial_hidden_state, np.ones((3, 2), dtype=np.float32))


def test_central_collection_setup_preserves_structured_packed_detection_contract() -> None:
    actors = [
        _actor(actor_id=0, supports_legal_candidate_scoring=True),
        _actor(actor_id=1, supports_legal_candidate_scoring=False),
    ]
    calls: list[int] = []

    def actor_inference_model(actor: Any) -> Any:
        calls.append(int(actor.actor_id))
        return actor

    assert supports_structured_central_packed(actors, actor_inference_model=actor_inference_model) is True
    assert calls == [0]

    non_packed = [_actor(actor_id=0, layout_name="dense")]
    assert supports_structured_central_packed(non_packed, actor_inference_model=actor_inference_model) is False


def test_central_collection_setup_rejects_empty_actor_list() -> None:
    with pytest.raises(ValueError, match="requires at least one actor"):
        build_central_actor_collection_setup(
            actors=[],
            config=SimpleNamespace(unroll_length=2, envs_per_actor=3),
            observation_dim=4,
            trajectory_retention_enabled=False,
            actor_inference_model=lambda actor: actor,
        )


def test_central_collection_falls_back_to_per_actor_collection_for_mixed_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actors = [_actor(actor_id=0, layout_name="i16_legal_ids"), _actor(actor_id=1, layout_name="dense")]
    runtime = SimpleNamespace(collected=[])

    def collect_one(actor: Any) -> str:
        runtime.collected.append(int(actor.actor_id))
        return f"unroll-{actor.actor_id}"

    def fail_setup(**_: Any) -> Any:
        raise AssertionError("mixed layouts should not build central setup")

    runtime._collect_actor_unroll = collect_one
    monkeypatch.setattr(central_collection_module, "build_central_actor_collection_setup", fail_setup)

    assert actors_have_single_layout(actors) is False
    assert QueueRuntimeCentralCollectionMixin._collect_actor_unrolls_central(runtime, actors) == [
        "unroll-0",
        "unroll-1",
    ]
    assert runtime.collected == [0, 1]
