from __future__ import annotations

from typing import Any

import torch
from weiss_rl.runtime.components.actors.actor_startup import build_runtime_actor_startup_state
from weiss_rl.runtime.components.config import QueueRuntimeConfig


class _DummyModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> _DummyModel:
        self.device = torch.device(device)
        return self

    def eval(self) -> _DummyModel:
        self.eval_called = True
        return self


def _config(*, actor_count: int = 2) -> QueueRuntimeConfig:
    return QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=actor_count,
        envs_per_actor=8,
        unroll_length=4,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=4,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1000,
    )


def test_actor_startup_builds_central_shared_model_and_local_actors() -> None:
    compiled: list[tuple[Any, bool]] = []
    actor_args: list[tuple[int, Any | None, Any | None]] = []

    state = build_runtime_actor_startup_state(
        model=_DummyModel(),
        config=_config(actor_count=2),
        process_actor_device_names=("cpu", "cpu"),
        actor_device=torch.device("cpu"),
        use_process_collectors=False,
        use_central_batched_collection=True,
        compile_actor_inference=True,
        build_actor_state=lambda actor_id, shared_model, compiled_model: (
            actor_args.append((actor_id, shared_model, compiled_model)) or {"actor_id": actor_id}
        ),
        maybe_compile_actor_model=lambda model, *, enabled: compiled.append((model, enabled)) or "compiled-model",
        actor_torch_threads=2,
        configure_actor_torch_threads=lambda threads: (_ for _ in ()).throw(
            AssertionError(f"unexpected thread config: {threads}")
        ),
    )

    assert isinstance(state.shared_actor_model, _DummyModel)
    assert state.shared_actor_model.device == torch.device("cpu")
    assert state.shared_actor_model.eval_called is True
    assert state.shared_compiled_actor_model == "compiled-model"
    assert compiled == [(state.shared_actor_model, True)]
    assert actor_args == [
        (0, state.shared_actor_model, state.shared_compiled_actor_model),
        (1, state.shared_actor_model, state.shared_compiled_actor_model),
    ]
    assert state.bootstrap_model_devices == []
    assert state.bootstrap_models is None
    assert state.actors == [{"actor_id": 0}, {"actor_id": 1}]
    assert state.collector_executor is None


def test_actor_startup_builds_process_bootstrap_models_without_local_actors() -> None:
    configured_threads: list[int] = []

    state = build_runtime_actor_startup_state(
        model=_DummyModel(),
        config=_config(actor_count=3),
        process_actor_device_names=("cpu", "cpu", "cpu"),
        actor_device=torch.device("cpu"),
        use_process_collectors=True,
        use_central_batched_collection=False,
        compile_actor_inference=False,
        build_actor_state=lambda actor_id, shared_model, compiled_model: (_ for _ in ()).throw(
            AssertionError(f"unexpected actor {actor_id}: {shared_model}, {compiled_model}")
        ),
        maybe_compile_actor_model=lambda model, *, enabled: (_ for _ in ()).throw(
            AssertionError(f"unexpected compile: {model}, {enabled}")
        ),
        actor_torch_threads=4,
        configure_actor_torch_threads=configured_threads.append,
    )

    assert state.shared_actor_model is None
    assert state.shared_compiled_actor_model is None
    assert state.bootstrap_model_devices == [torch.device("cpu"), torch.device("cpu"), torch.device("cpu")]
    assert state.bootstrap_models is not None
    assert [model.device for model in state.bootstrap_models] == [torch.device("cpu")] * 3
    assert state.actors == []
    assert state.collector_executor is None
    assert configured_threads == []


def test_actor_startup_builds_thread_pool_for_multiple_local_actors() -> None:
    configured_threads: list[int] = []

    state = build_runtime_actor_startup_state(
        model=_DummyModel(),
        config=_config(actor_count=2),
        process_actor_device_names=("cpu", "cpu"),
        actor_device=torch.device("cpu"),
        use_process_collectors=False,
        use_central_batched_collection=False,
        compile_actor_inference=False,
        build_actor_state=lambda actor_id, shared_model, compiled_model: {
            "actor_id": actor_id,
            "shared_model": shared_model,
            "compiled_model": compiled_model,
        },
        maybe_compile_actor_model=lambda model, *, enabled: None,
        actor_torch_threads=3,
        configure_actor_torch_threads=configured_threads.append,
    )

    try:
        assert state.shared_actor_model is None
        assert state.shared_compiled_actor_model is None
        assert state.bootstrap_model_devices == []
        assert state.bootstrap_models is None
        assert state.actors == [
            {"actor_id": 0, "shared_model": None, "compiled_model": None},
            {"actor_id": 1, "shared_model": None, "compiled_model": None},
        ]
        assert state.collector_executor is not None
        assert configured_threads == [3]
    finally:
        if state.collector_executor is not None:
            state.collector_executor.shutdown(wait=True)
