from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from weiss_rl.runtime import QueueRuntime, resolve_actor_device_layout

from .runtime_topology_test_support import (
    async_runtime_config,
    patch_fake_actor_state_builder,
    runtime_stack,
)


def test_runtime_honors_non_cpu_actor_device_and_disables_process_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: True)
    patch_fake_actor_state_builder(monkeypatch)
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=runtime_stack(actor_device="cuda:0", mixed_precision=True),
        config=async_runtime_config(),
        model=cast(Any, object()),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._device == torch.device("cuda:0")
        assert runtime_any._actor_amp_enabled is True
        assert runtime_any._use_process_collectors is False
    finally:
        runtime.close()


def test_resolve_actor_device_layout_spreads_cuda_auto_across_non_learner_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.runtime.components.devices.torch.cuda.device_count", lambda: 4)

    layout = resolve_actor_device_layout(
        runtime_stack(actor_device="cuda:auto", learner_device="cuda:auto"),
        actor_count=5,
        learner_device=torch.device("cuda:0"),
        prefer_process_collectors=True,
    )

    assert layout == ("cuda:1", "cuda:2", "cuda:3", "cuda:1", "cuda:2")
