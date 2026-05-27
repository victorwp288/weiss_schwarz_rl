from __future__ import annotations

from typing import Any

import torch


def is_cuda_auto_request(requested: str) -> bool:
    normalized = str(requested).strip().lower()
    return normalized in {"auto", "cuda:auto", "cuda:all", "all"}


def available_cuda_device_names() -> tuple[str, ...]:
    if not torch.cuda.is_available():
        return ()
    return tuple(f"cuda:{index}" for index in range(int(torch.cuda.device_count())))


def normalize_device_name(requested: str) -> str:
    value = str(requested).strip()
    if not value:
        return "cpu"
    if is_cuda_auto_request(value):
        available = available_cuda_device_names()
        return "cpu" if not available else available[0]
    if value.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else int(device.index)}"
    return str(device)


def configured_learner_device_name(
    stack: Any,
    *,
    learner_device: torch.device | str | None = None,
) -> str:
    if learner_device is not None:
        return normalize_device_name(str(learner_device))
    system = stack.config.system
    requested = "cpu" if system is None else str(getattr(system, "learner_device", "cpu")).strip()
    return normalize_device_name(requested)


def resolve_actor_device_layout(
    stack: Any,
    *,
    actor_count: int,
    learner_device: torch.device | str | None = None,
    prefer_process_collectors: bool = False,
) -> tuple[str, ...]:
    count = max(1, int(actor_count))
    system = stack.config.system
    requested = "cpu" if system is None else str(getattr(system, "actor_device", "cpu")).strip()
    if not requested:
        requested = "cpu"
    requested_parts = tuple(part.strip() for part in requested.split(",") if part.strip())
    if len(requested_parts) > 1:
        normalized_parts = tuple(normalize_device_name(part) for part in requested_parts)
        return tuple(normalized_parts[index % len(normalized_parts)] for index in range(count))
    if is_cuda_auto_request(requested):
        available = available_cuda_device_names()
        if not available:
            return ("cpu",) * count
        learner_name = configured_learner_device_name(stack, learner_device=learner_device)
        actor_pool = tuple(device_name for device_name in available if device_name != learner_name)
        if not actor_pool:
            actor_pool = available
        if not prefer_process_collectors:
            return (actor_pool[0],) * count
        return tuple(actor_pool[index % len(actor_pool)] for index in range(count))
    normalized = normalize_device_name(requested)
    return (normalized,) * count
