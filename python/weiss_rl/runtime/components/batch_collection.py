"""Shared pending-unroll collection lifecycle for queue-runtime learner batches."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from weiss_rl.runtime.components.types import PendingUnroll, RuntimeBatch

RuntimeBatchBuilder = Callable[[Sequence[PendingUnroll]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PendingUnrollKey:
    actor_id: int
    unroll_seq: int


@dataclass(frozen=True, slots=True)
class PendingUnrollSelection:
    selected: tuple[PendingUnroll, ...]
    removed_keys: frozenset[PendingUnrollKey]


@dataclass(frozen=True, slots=True)
class BatchCollectionSpec:
    target_count: int
    build_batch: RuntimeBatchBuilder
    build_timer_name: str
    total_timer_name: str


def pending_unroll_key(unroll: PendingUnroll) -> PendingUnrollKey:
    return PendingUnrollKey(actor_id=int(unroll.actor_id), unroll_seq=int(unroll.unroll_seq))


def pending_unroll_selection(selected: Sequence[PendingUnroll]) -> PendingUnrollSelection:
    selected_tuple = tuple(selected)
    return PendingUnrollSelection(
        selected=selected_tuple,
        removed_keys=frozenset(pending_unroll_key(item) for item in selected_tuple),
    )


def remaining_pending_unrolls(
    pending_unrolls: Iterable[PendingUnroll],
    selection: PendingUnrollSelection,
) -> deque[PendingUnroll]:
    return deque(item for item in pending_unrolls if pending_unroll_key(item) not in selection.removed_keys)


def select_and_remove_pending_unrolls(runtime: Any) -> PendingUnrollSelection:
    selection = pending_unroll_selection(runtime._select_pending_unrolls())
    runtime._pending_unrolls = remaining_pending_unrolls(runtime._pending_unrolls, selection)
    return selection


def _log_runtime_performance(runtime: Any, runtime_metrics: dict[str, float]) -> None:
    if runtime._performance_logger is None:
        return
    elapsed = time.time() - runtime._runtime_start
    log_started = time.perf_counter()
    runtime._performance_logger.log(
        {
            "kind": "runtime_performance_v1",
            "wall_clock_seconds": elapsed,
            **runtime_metrics,
            **runtime._batch_timer_metrics,
        }
    )
    runtime._record_batch_timer_ms("performance_log", time.perf_counter() - log_started)


def _fill_pending_unrolls_for_batch(runtime: Any, *, target_count: int) -> list[float]:
    occupancy_samples: list[float] = []
    fill_started = time.perf_counter()
    runtime._fill_pending_unrolls(
        target_count=int(target_count),
        occupancy_samples=occupancy_samples,
    )
    runtime._record_batch_timer_ms("fill_pending_unrolls", time.perf_counter() - fill_started)
    return occupancy_samples


def _build_selected_learner_batch(
    runtime: Any,
    selection: PendingUnrollSelection,
    *,
    spec: BatchCollectionSpec,
    occupancy_samples: Sequence[float],
) -> tuple[dict[str, Any], dict[str, float]]:
    selected = list(selection.selected)
    build_started = time.perf_counter()
    try:
        learner_batch = spec.build_batch(selected)
        runtime._record_batch_timer_ms(spec.build_timer_name, time.perf_counter() - build_started)
        runtime_metrics = runtime._runtime_metrics(selected, occupancy_samples=occupancy_samples)
        return learner_batch, runtime_metrics
    finally:
        runtime._release_shared_pending_unrolls(selected)


def collect_pending_runtime_batch(
    runtime: Any,
    *,
    target_count: int,
    build_batch: RuntimeBatchBuilder,
    build_timer_name: str,
    total_timer_name: str,
) -> RuntimeBatch:
    spec = BatchCollectionSpec(
        target_count=int(target_count),
        build_batch=build_batch,
        build_timer_name=str(build_timer_name),
        total_timer_name=str(total_timer_name),
    )
    batch_started = time.perf_counter()
    runtime._reset_batch_timer_metrics()
    occupancy_samples = _fill_pending_unrolls_for_batch(runtime, target_count=spec.target_count)
    selection = select_and_remove_pending_unrolls(runtime)

    learner_batch, runtime_metrics = _build_selected_learner_batch(
        runtime,
        selection,
        spec=spec,
        occupancy_samples=occupancy_samples,
    )
    runtime._record_batch_timer_ms(spec.total_timer_name, time.perf_counter() - batch_started)
    _log_runtime_performance(runtime, runtime_metrics)
    runtime_metrics.update(runtime._batch_timer_metrics)
    return RuntimeBatch(learner_batch=learner_batch, runtime_metrics=runtime_metrics)


__all__ = [
    "BatchCollectionSpec",
    "PendingUnrollKey",
    "PendingUnrollSelection",
    "RuntimeBatchBuilder",
    "collect_pending_runtime_batch",
    "pending_unroll_key",
    "pending_unroll_selection",
    "remaining_pending_unrolls",
    "select_and_remove_pending_unrolls",
]
