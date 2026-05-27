from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import torch


def configure_torch_threads(stack: Any) -> None:
    system_config = stack.config.system
    if system_config is None:
        return
    torch.set_num_threads(int(system_config.learner_torch_threads))
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


@contextmanager
def torch_num_threads_scope(num_threads: int | None) -> Iterator[None]:
    if num_threads is None:
        yield
        return
    target = int(num_threads)
    if target < 1:
        raise ValueError("num_threads must be >= 1")
    previous = int(torch.get_num_threads())
    if previous == target:
        yield
        return
    torch.set_num_threads(target)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def central_runtime_actor_torch_threads(stack: Any, runtime: Any) -> int | None:
    system_config = stack.config.system
    if system_config is None:
        return None
    if str(system_config.actor_device).strip().lower() != "cpu":
        return None
    if bool(getattr(runtime, "_use_process_collectors", False)):
        return None
    if not bool(getattr(runtime, "_use_central_batched_collection", False)):
        return None
    return int(system_config.actor_torch_threads)
