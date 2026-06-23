from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from weiss_rl.runtime import QueueRuntime

from .runtime_topology_test_support import (
    DummyDeviceModel,
    DummyProcessModel,
    async_runtime_config,
    patch_fake_actor_state_builder,
    runtime_stack,
)


def test_runtime_can_force_process_collectors_for_structured_cuda_auto_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started_with: list[Any] = []

    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: started_with.append(model),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = DummyProcessModel()
    runtime = QueueRuntime(
        stack=runtime_stack(
            actor_device="cuda:auto",
            learner_device="cuda:auto",
            collection_backend="process",
            mixed_precision=True,
            structured_warmstart_enabled=True,
            role="main",
            league_enabled=True,
            encoder_kind="structured_v2",
        ),
        config=async_runtime_config(actor_count=4),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
        learner_device=torch.device("cuda:0"),
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._collection_backend == "process"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert runtime_any._process_actor_device_names == ("cuda:1", "cuda:2", "cuda:3", "cuda:1")
        assert started_with == [dummy_model]
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_typed_cpu_async(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: False)
    patch_fake_actor_state_builder(monkeypatch)
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=runtime_stack(actor_device="cpu", encoder_kind="typed_v1"),
        config=async_runtime_config(),
        model=cast(Any, torch.nn.Linear(8, 4)),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_auto_prefers_process_collectors_for_structured_cpu_model_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_with: list[Any] = []

    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: started_with.append(model),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=runtime_stack(
            actor_device="cpu",
            actor_policy_backend="model",
            structured_warmstart_enabled=False,
            encoder_kind="structured_v2",
        ),
        config=async_runtime_config(),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._collection_backend == "auto"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert started_with == [dummy_model]
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_structured_cuda_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: True)
    patch_fake_actor_state_builder(monkeypatch)
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=runtime_stack(
            actor_device="cuda:0",
            mixed_precision=True,
            structured_warmstart_enabled=False,
            encoder_kind="structured_v2",
        ),
        config=async_runtime_config(),
        model=cast(Any, DummyDeviceModel()),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._device == torch.device("cuda:0")
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_keeps_central_batched_collection_for_typed_cpu_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: False)
    patch_fake_actor_state_builder(monkeypatch)
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=runtime_stack(actor_device="cpu", role="main", league_enabled=True, encoder_kind="typed_v1"),
        config=async_runtime_config(),
        model=cast(Any, torch.nn.Linear(8, 4)),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_can_force_process_collectors_for_structured_cpu_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started_with: list[Any] = []

    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: started_with.append(model),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=runtime_stack(
            actor_device="cpu",
            collection_backend="process",
            structured_warmstart_enabled=True,
            role="main",
            league_enabled=True,
            encoder_kind="structured_v2",
        ),
        config=async_runtime_config(),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._collection_backend == "process"
        assert runtime_any._use_central_batched_collection is False
        assert runtime_any._use_process_collectors is True
        assert started_with == [dummy_model]
    finally:
        runtime.close()


def test_runtime_process_collectors_start_before_refreshing_opponent_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_order: list[str] = []

    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_start_process_collectors",
        lambda self, model: call_order.append("start"),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: call_order.append("refresh"))

    runtime = QueueRuntime(
        stack=runtime_stack(
            actor_device="cpu",
            collection_backend="process",
            structured_warmstart_enabled=True,
            role="main",
            league_enabled=True,
            encoder_kind="structured_v2",
        ),
        config=async_runtime_config(),
        model=cast(Any, torch.nn.Linear(8, 4)),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        assert call_order == ["start", "refresh"]
    finally:
        runtime.close()
