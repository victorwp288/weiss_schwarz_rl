"""Training profiler helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch


@contextmanager
def profile_block(enabled: bool, name: str) -> Iterator[None]:
    """Record a named autograd profiler block when timer profiling is enabled."""

    if not enabled:
        yield
        return
    with torch.autograd.profiler.record_function(name):
        yield


def build_training_profiler(
    *,
    enabled: bool,
    run_dir: Path,
    device: torch.device,
) -> tuple[Any | None, Any, Path | None]:
    """Build the optional torch profiler and trace directory for a training run."""

    if not enabled:
        return None, nullcontext(), None

    profile_dir = run_dir / "profiling" / "torch_profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    return profiler, profiler, profile_dir
