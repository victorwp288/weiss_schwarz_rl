from __future__ import annotations

from types import SimpleNamespace

import pytest

from weiss_rl.training import torch_threads


def test_torch_num_threads_scope_restores_previous_value(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"threads": 8}
    calls: list[int] = []

    monkeypatch.setattr(torch_threads.torch, "get_num_threads", lambda: state["threads"])

    def _set_num_threads(value: int) -> None:
        state["threads"] = int(value)
        calls.append(int(value))

    monkeypatch.setattr(torch_threads.torch, "set_num_threads", _set_num_threads)

    with torch_threads.torch_num_threads_scope(2):
        assert state["threads"] == 2

    assert state["threads"] == 8
    assert calls == [2, 8]


def test_torch_num_threads_scope_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="num_threads must be >= 1"):
        with torch_threads.torch_num_threads_scope(0):
            pass


def test_central_runtime_actor_torch_threads_requires_cpu_central_runtime() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(system=SimpleNamespace(actor_device="cpu", actor_torch_threads=16)))
    central_runtime = SimpleNamespace(_use_process_collectors=False, _use_central_batched_collection=True)
    process_runtime = SimpleNamespace(_use_process_collectors=True, _use_central_batched_collection=False)

    assert torch_threads.central_runtime_actor_torch_threads(stack, central_runtime) == 16
    assert torch_threads.central_runtime_actor_torch_threads(stack, process_runtime) is None
