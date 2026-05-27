"""Training manifest layout helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from weiss_rl.config import StackConfig
from weiss_rl.runtime import QueueRuntimeMode, build_runtime_config, resolve_actor_device_layout


def manifest_actor_device_layout(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    learner_device: torch.device,
) -> tuple[str, ...] | None:
    if stack.config.system is None or stack.config.training is None:
        return None
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
    )
    return tuple(
        str(device_name)
        for device_name in resolve_actor_device_layout(
            stack,
            actor_count=int(runtime_config.actor_count),
            learner_device=learner_device,
            prefer_process_collectors=True,
        )
    )


def hardware_actor_layout_payload(actor_device_layout: Sequence[Any] | None) -> tuple[str, ...] | None:
    if not actor_device_layout:
        return None
    return tuple(str(device_name) for device_name in actor_device_layout)
