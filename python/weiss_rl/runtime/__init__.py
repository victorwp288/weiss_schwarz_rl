"""Public runtime surface for queue-based Weiss Schwarz training."""

from __future__ import annotations

from weiss_rl.runtime.components.config import QueueRuntimeConfig, build_runtime_config
from weiss_rl.runtime.components.devices import resolve_actor_device_layout
from weiss_rl.runtime.components.topology import QueueRuntimeMode
from weiss_rl.runtime.components.types import RuntimeBatch, RuntimeUnroll
from weiss_rl.runtime.queue_runtime import QueueRuntime

__all__ = [
    "QueueRuntime",
    "QueueRuntimeConfig",
    "QueueRuntimeMode",
    "RuntimeBatch",
    "RuntimeUnroll",
    "build_runtime_config",
    "resolve_actor_device_layout",
]
