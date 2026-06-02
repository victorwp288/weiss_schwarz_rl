from __future__ import annotations

import pytest

from weiss_rl.runtime import _configure_runtime_actor_torch_threads
from weiss_rl.runtime.components import threads as runtime_threads


def test_configure_runtime_actor_torch_threads_preserves_runtime_wrapper(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(runtime_threads.torch, "set_num_threads", lambda value: calls.append(("threads", int(value))))
    monkeypatch.setattr(
        runtime_threads.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", int(value))),
    )

    runtime_threads.configure_runtime_actor_torch_threads(3)
    _configure_runtime_actor_torch_threads(4)

    assert _configure_runtime_actor_torch_threads is not runtime_threads.configure_runtime_actor_torch_threads
    assert calls == [("threads", 3), ("interop", 1), ("threads", 4), ("interop", 1)]


def test_configure_runtime_actor_torch_threads_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="actor_torch_threads must be >= 1"):
        runtime_threads.configure_runtime_actor_torch_threads(0)


def test_configure_runtime_actor_torch_threads_suppresses_interop_runtime_error(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def _raise_interop(_value: int) -> None:
        raise RuntimeError("already configured")

    monkeypatch.setattr(runtime_threads.torch, "set_num_threads", lambda value: calls.append(("threads", int(value))))
    monkeypatch.setattr(runtime_threads.torch, "set_num_interop_threads", _raise_interop)

    runtime_threads.configure_runtime_actor_torch_threads(2)

    assert calls == [("threads", 2)]
