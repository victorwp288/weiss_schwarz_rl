"""Training profiling helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch


@contextmanager
def profile_block(enabled: bool, name: str) -> Iterator[None]:
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
) -> tuple[torch.profiler.profile | None, Any, Path | None]:
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


def format_torch_profiler_trace_written_message(trace_path: Path) -> str:
    return f"Wrote torch profiler trace: {trace_path}"
