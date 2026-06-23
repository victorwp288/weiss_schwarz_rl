"""Collection-backend selection for QueueRuntime startup."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from weiss_rl.runtime.components.config import QueueRuntimeConfig
from weiss_rl.runtime.components.devices import is_cuda_auto_request


@dataclass(frozen=True)
class RuntimeCollectionBackendSelection:
    collection_backend: str
    use_process_collectors: bool
    use_central_batched_collection: bool


def select_runtime_collection_backend(
    *,
    config: QueueRuntimeConfig,
    requested_collection_backend: str,
    requested_actor_device: str,
    process_actor_device_names: tuple[str, ...],
    actor_device: torch.device,
    actor_policy_backend: str,
    model_kind: str,
    league_enabled: bool,
) -> RuntimeCollectionBackendSelection:
    collection_backend = str(requested_collection_backend).strip().lower()
    if collection_backend not in {"auto", "central", "process"}:
        raise ValueError("system.collection_backend must be one of: auto, central, process")

    process_collectors_supported = bool(
        config.mode == "train_async_fast"
        and int(config.actor_count) > 1
        and model_kind != "typed_v1"
        and all(torch.device(device_name).type in {"cpu", "cuda"} for device_name in process_actor_device_names)
    )
    auto_use_process_collectors = bool(process_collectors_supported and not league_enabled)
    central_batched_collection_supported = bool(
        config.mode == "train_async_fast"
        and (
            (actor_device.type == "cpu" and model_kind in {"typed_v1", "structured_v2"})
            or (actor_device.type == "cuda" and model_kind == "structured_v2")
        )
    )
    auto_use_central_batched_collection = bool(central_batched_collection_supported)
    auto_prefers_process_collectors = bool(
        auto_use_process_collectors
        and (
            (actor_policy_backend == "model" and actor_device.type == "cpu" and model_kind == "structured_v2")
            or (
                (is_cuda_auto_request(requested_actor_device) or "," in requested_actor_device)
                and len(dict.fromkeys(process_actor_device_names)) > 1
            )
        )
    )
    if auto_prefers_process_collectors:
        auto_use_central_batched_collection = False
    elif auto_use_central_batched_collection:
        auto_use_process_collectors = False

    if collection_backend == "auto":
        return RuntimeCollectionBackendSelection(
            collection_backend=collection_backend,
            use_process_collectors=auto_use_process_collectors,
            use_central_batched_collection=auto_use_central_batched_collection,
        )
    if collection_backend == "central":
        if not central_batched_collection_supported:
            raise ValueError("system.collection_backend=central is not supported for the current runtime setup")
        return RuntimeCollectionBackendSelection(
            collection_backend=collection_backend,
            use_process_collectors=False,
            use_central_batched_collection=True,
        )
    if not process_collectors_supported:
        raise ValueError("system.collection_backend=process is not supported for the current runtime setup")
    return RuntimeCollectionBackendSelection(
        collection_backend=collection_backend,
        use_process_collectors=True,
        use_central_batched_collection=False,
    )


__all__ = ["RuntimeCollectionBackendSelection", "select_runtime_collection_backend"]
