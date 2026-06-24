"""Actor model and local collector startup state for QueueRuntime."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from weiss_rl.runtime.components.config import QueueRuntimeConfig


@dataclass(frozen=True)
class RuntimeActorStartupState:
    bootstrap_model_devices: list[torch.device]
    shared_actor_model: Any | None
    shared_compiled_actor_model: Any | None
    bootstrap_models: list[Any] | None
    actors: list[Any]
    collector_executor: ThreadPoolExecutor | None


def build_runtime_actor_startup_state(
    *,
    model: Any,
    config: QueueRuntimeConfig,
    process_actor_device_names: Sequence[str],
    actor_device: torch.device,
    use_process_collectors: bool,
    use_central_batched_collection: bool,
    compile_actor_inference: bool,
    build_actor_state: Callable[[int, Any | None, Any | None], Any],
    maybe_compile_actor_model: Callable[..., Any | None],
    actor_torch_threads: int | None,
    configure_actor_torch_threads: Callable[[int], None],
) -> RuntimeActorStartupState:
    bootstrap_model_devices = (
        [torch.device(device_name) for device_name in process_actor_device_names] if use_process_collectors else []
    )
    shared_actor_model = None
    shared_compiled_actor_model = None
    if use_central_batched_collection:
        shared_actor_model = copy.deepcopy(model).to(actor_device)
        shared_actor_model.eval()
        shared_compiled_actor_model = maybe_compile_actor_model(
            shared_actor_model,
            enabled=bool(compile_actor_inference),
        )

    bootstrap_models = (
        [copy.deepcopy(model).to(bootstrap_model_devices[actor_id]) for actor_id in range(int(config.actor_count))]
        if use_process_collectors
        else None
    )
    actors = (
        []
        if use_process_collectors
        else [
            build_actor_state(actor_id, shared_actor_model, shared_compiled_actor_model)
            for actor_id in range(int(config.actor_count))
        ]
    )
    collector_executor = (
        None
        if use_process_collectors or use_central_batched_collection or len(actors) <= 1
        else ThreadPoolExecutor(
            max_workers=len(actors),
            thread_name_prefix="weiss-runtime-actor",
        )
    )
    if collector_executor is not None and actor_torch_threads is not None:
        configure_actor_torch_threads(int(actor_torch_threads))

    return RuntimeActorStartupState(
        bootstrap_model_devices=bootstrap_model_devices,
        shared_actor_model=shared_actor_model,
        shared_compiled_actor_model=shared_compiled_actor_model,
        bootstrap_models=bootstrap_models,
        actors=actors,
        collector_executor=collector_executor,
    )


__all__ = ["RuntimeActorStartupState", "build_runtime_actor_startup_state"]
