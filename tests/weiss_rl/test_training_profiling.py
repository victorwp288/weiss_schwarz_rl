from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from weiss_rl.training import profiling
from weiss_rl.training.profiling import build_training_profiler, profile_block


class _RecordingContext:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self) -> None:
        self._events.append(f"enter:{self._name}")

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self._events.append(f"exit:{self._name}")


def test_profile_block_disabled_does_not_record(monkeypatch: Any) -> None:
    def _fail_record_function(name: str) -> _RecordingContext:
        raise AssertionError(f"record_function should not be called for {name}")

    monkeypatch.setattr(profiling.torch.autograd.profiler, "record_function", _fail_record_function)

    with profile_block(False, "collect_update_batch"):
        pass


def test_profile_block_enabled_records_named_region(monkeypatch: Any) -> None:
    events: list[str] = []

    def _record_function(name: str) -> _RecordingContext:
        return _RecordingContext(events, name)

    monkeypatch.setattr(profiling.torch.autograd.profiler, "record_function", _record_function)

    with profile_block(True, "learner_update"):
        events.append("inside")

    assert events == ["enter:learner_update", "inside", "exit:learner_update"]


def test_build_training_profiler_disabled_returns_null_context_without_directory(tmp_path: Path) -> None:
    profiler, profiler_context, trace_dir = build_training_profiler(
        enabled=False,
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert profiler is None
    assert trace_dir is None
    assert not (tmp_path / "profiling").exists()
    with profiler_context:
        pass


def test_build_training_profiler_enabled_creates_cpu_profiler(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    fake_profiler = object()

    def _profile(**kwargs: Any) -> object:
        calls.append(kwargs)
        return fake_profiler

    monkeypatch.setattr(profiling.torch.profiler, "profile", _profile)

    profiler, profiler_context, trace_dir = build_training_profiler(
        enabled=True,
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert profiler is fake_profiler
    assert profiler_context is fake_profiler
    assert trace_dir == tmp_path / "profiling" / "torch_profiler"
    assert trace_dir.is_dir()
    assert calls == [
        {
            "activities": [torch.profiler.ProfilerActivity.CPU],
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }
    ]


def test_build_training_profiler_includes_cuda_activity_for_cuda_device(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def _profile(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(profiling.torch.profiler, "profile", _profile)

    build_training_profiler(
        enabled=True,
        run_dir=tmp_path,
        device=torch.device("cuda"),
    )

    assert calls[0]["activities"] == [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
