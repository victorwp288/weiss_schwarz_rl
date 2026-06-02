"""Torch thread configuration helpers for runtime actors."""

from __future__ import annotations

from contextlib import suppress

import torch


def configure_runtime_actor_torch_threads(actor_torch_threads: int) -> None:
    threads = int(actor_torch_threads)
    if threads < 1:
        raise ValueError("actor_torch_threads must be >= 1")
    torch.set_num_threads(threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
