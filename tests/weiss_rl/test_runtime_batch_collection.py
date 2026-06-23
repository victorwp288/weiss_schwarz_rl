from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime import queue_runtime as queue_runtime_module
from weiss_rl.runtime.components.batch_collection import (
    collect_pending_runtime_batch,
    pending_unroll_key,
    pending_unroll_selection,
    remaining_pending_unrolls,
)
from weiss_rl.runtime.components.types import RuntimeBatch


def _unroll(actor_id: int, unroll_seq: int) -> SimpleNamespace:
    return SimpleNamespace(actor_id=actor_id, unroll_seq=unroll_seq, behavior_policy_version=0)


class _PerformanceLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, float | str]] = []

    def log(self, record: dict[str, float | str]) -> None:
        self.records.append(record)


class _Runtime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(batch_unrolls_per_update=2)
        self._pending_unrolls = deque([_unroll(0, 0), _unroll(1, 0), _unroll(2, 0)])
        self._batch_timer_metrics: dict[str, float] = {"stale": 99.0}
        self._performance_logger = _PerformanceLogger()
        self._runtime_start = time.time() - 5.0
        self.events: list[tuple[str, Any]] = []

    def _reset_batch_timer_metrics(self) -> None:
        self.events.append(("reset_timers", None))
        self._batch_timer_metrics = {}

    def _fill_pending_unrolls(self, *, target_count: int, occupancy_samples: list[float]) -> None:
        self.events.append(("fill", target_count))
        occupancy_samples.extend([0.25, 0.75])

    def _record_batch_timer_ms(self, name: str, elapsed_seconds: float) -> None:
        self.events.append(("timer", name))
        self._batch_timer_metrics[f"runtime_{name}"] = self._batch_timer_metrics.get(
            f"runtime_{name}",
            0.0,
        ) + max(0.0, float(elapsed_seconds) * 1000.0)

    def _select_pending_unrolls(self) -> list[Any]:
        self.events.append(("select", None))
        return list(self._pending_unrolls)[:2]

    def _runtime_metrics(self, selected: list[Any], *, occupancy_samples: list[float]) -> dict[str, float]:
        self.events.append(("metrics", [(item.actor_id, item.unroll_seq) for item in selected], occupancy_samples))
        return {"batch_env_steps": float(len(selected))}

    def _release_shared_pending_unrolls(self, selected: list[Any]) -> None:
        self.events.append(("release", [(item.actor_id, item.unroll_seq) for item in selected]))


def test_collect_pending_runtime_batch_preserves_selection_timing_logging_and_release_order() -> None:
    runtime = _Runtime()

    def build_batch(selected: list[Any]) -> dict[str, object]:
        runtime.events.append(("build", [(item.actor_id, item.unroll_seq) for item in selected]))
        assert [(item.actor_id, item.unroll_seq) for item in runtime._pending_unrolls] == [(2, 0)]
        return {"selected": tuple((item.actor_id, item.unroll_seq) for item in selected)}

    batch = collect_pending_runtime_batch(
        runtime,
        target_count=2,
        build_batch=build_batch,
        build_timer_name="build_learner_batch",
        total_timer_name="collect_update_batch_total",
    )

    assert batch.learner_batch == {"selected": ((0, 0), (1, 0))}
    assert [(item.actor_id, item.unroll_seq) for item in runtime._pending_unrolls] == [(2, 0)]
    assert batch.runtime_metrics["batch_env_steps"] == pytest.approx(2.0)
    assert "runtime_fill_pending_unrolls" in batch.runtime_metrics
    assert "runtime_build_learner_batch" in batch.runtime_metrics
    assert "runtime_collect_update_batch_total" in batch.runtime_metrics
    assert "runtime_performance_log" in batch.runtime_metrics
    assert [event[0] for event in runtime.events] == [
        "reset_timers",
        "fill",
        "timer",
        "select",
        "build",
        "timer",
        "metrics",
        "release",
        "timer",
        "timer",
    ]
    assert runtime.events[6] == ("metrics", [(0, 0), (1, 0)], [0.25, 0.75])
    assert runtime.events[7] == ("release", [(0, 0), (1, 0)])
    assert runtime._performance_logger.records
    logged = runtime._performance_logger.records[0]
    assert logged["kind"] == "runtime_performance_v1"
    assert logged["batch_env_steps"] == pytest.approx(2.0)
    assert "runtime_collect_update_batch_total" in logged
    assert "runtime_performance_log" not in logged


def test_pending_unroll_selection_removes_matching_keys_and_preserves_remaining_order() -> None:
    first = _unroll(0, 0)
    selected_tail = _unroll(1, 1)
    duplicate_key = _unroll(0, 0)
    pending = deque([first, _unroll(1, 0), duplicate_key, _unroll(2, 0), selected_tail])

    selection = pending_unroll_selection([first, selected_tail])
    remaining = remaining_pending_unrolls(pending, selection)

    assert selection.selected == (first, selected_tail)
    assert selection.removed_keys == {
        pending_unroll_key(first),
        pending_unroll_key(selected_tail),
    }
    assert [(item.actor_id, item.unroll_seq) for item in remaining] == [(1, 0), (2, 0)]


def test_collect_pending_runtime_batch_releases_selected_unrolls_when_builder_fails() -> None:
    runtime = _Runtime()

    def fail_build(_selected: list[Any]) -> dict[str, object]:
        runtime.events.append(("build", "fail"))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        collect_pending_runtime_batch(
            runtime,
            target_count=2,
            build_batch=fail_build,
            build_timer_name="build_learner_batch",
            total_timer_name="collect_update_batch_total",
        )

    assert [event[0] for event in runtime.events] == [
        "reset_timers",
        "fill",
        "timer",
        "select",
        "build",
        "release",
    ]
    assert runtime.events[-1] == ("release", [(0, 0), (1, 0)])


def test_queue_runtime_collect_update_batch_threads_impala_builder_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime.config = SimpleNamespace(batch_unrolls_per_update=7)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_builder(selected: list[object], **kwargs: object) -> dict[str, object]:
        calls.append(("build", {"selected": selected, **kwargs}))
        return {"kind": "impala"}

    def fake_collect(self: object, **kwargs: object) -> RuntimeBatch:
        calls.append(("collect", dict(kwargs)))
        learner_batch = kwargs["build_batch"](["u0", "u1"])
        return RuntimeBatch(learner_batch=learner_batch, runtime_metrics={"runtime": 1.0})

    runtime._build_learner_batch = fake_builder
    monkeypatch.setattr(queue_runtime_module, "collect_pending_runtime_batch", fake_collect)

    batch = QueueRuntime.collect_update_batch(
        runtime,
        gamma=0.9,
        truncation_reward=-0.5,
        truncation_bootstrap_value=True,
        vtrace_rho_bar=1.25,
        vtrace_c_bar=0.75,
    )

    assert batch.learner_batch == {"kind": "impala"}
    assert batch.runtime_metrics == {"runtime": 1.0}
    assert calls[0] == (
        "collect",
        {
            "target_count": 7,
            "build_batch": calls[0][1]["build_batch"],
            "build_timer_name": "build_learner_batch",
            "total_timer_name": "collect_update_batch_total",
        },
    )
    assert calls[1] == (
        "build",
        {
            "selected": ["u0", "u1"],
            "gamma": 0.9,
            "truncation_reward": -0.5,
            "truncation_bootstrap_value": True,
            "vtrace_rho_bar": 1.25,
            "vtrace_c_bar": 0.75,
        },
    )


def test_queue_runtime_collect_policy_batch_threads_ppo_builder_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime.config = SimpleNamespace(batch_unrolls_per_update=5)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_builder(selected: list[object], **kwargs: object) -> dict[str, object]:
        calls.append(("build", {"selected": selected, **kwargs}))
        return {"kind": "ppo"}

    def fake_collect(self: object, **kwargs: object) -> RuntimeBatch:
        calls.append(("collect", dict(kwargs)))
        learner_batch = kwargs["build_batch"](["u2"])
        return RuntimeBatch(learner_batch=learner_batch, runtime_metrics={"runtime": 2.0})

    runtime._build_ppo_batch = fake_builder
    monkeypatch.setattr(queue_runtime_module, "collect_pending_runtime_batch", fake_collect)

    batch = QueueRuntime.collect_policy_batch(
        runtime,
        gamma=0.95,
        gae_lambda=0.8,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
    )

    assert batch.learner_batch == {"kind": "ppo"}
    assert batch.runtime_metrics == {"runtime": 2.0}
    assert calls[0] == (
        "collect",
        {
            "target_count": 5,
            "build_batch": calls[0][1]["build_batch"],
            "build_timer_name": "build_ppo_batch",
            "total_timer_name": "collect_policy_batch_total",
        },
    )
    assert calls[1] == (
        "build",
        {
            "selected": ["u2"],
            "gamma": 0.95,
            "gae_lambda": 0.8,
            "truncation_reward": -0.25,
            "truncation_bootstrap_value": False,
        },
    )
