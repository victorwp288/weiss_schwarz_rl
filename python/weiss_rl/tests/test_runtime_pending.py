from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace

import pytest

from weiss_rl.runtime_components.pending import (
    diverse_batch_target_count,
    pending_diverse_unroll_count,
    select_pending_unrolls,
)
from weiss_rl.runtime_components.pending_mixin import QueueRuntimePendingMixin


def _pending(*, actor_id: int, unroll_seq: int, behavior_policy_version: int) -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
    )


def test_select_pending_unrolls_train_ordered_keeps_oldest_same_sequence_group() -> None:
    pending = [
        _pending(actor_id=2, unroll_seq=2, behavior_policy_version=1),
        _pending(actor_id=0, unroll_seq=1, behavior_policy_version=1),
        _pending(actor_id=1, unroll_seq=1, behavior_policy_version=1),
        _pending(actor_id=3, unroll_seq=1, behavior_policy_version=2),
    ]

    selected = select_pending_unrolls(
        pending,
        batch_size=2,
        mode="train_ordered",
        diverse_opponent_actor_count=0,
        diverse_opponent_batch_fraction=0.0,
    )

    assert [(item.actor_id, item.unroll_seq, item.behavior_policy_version) for item in selected] == [
        (0, 1, 1),
        (1, 1, 1),
    ]


def test_select_pending_unrolls_train_ordered_rejects_empty_queue() -> None:
    with pytest.raises(RuntimeError, match="requires at least one pending unroll"):
        select_pending_unrolls(
            [],
            batch_size=2,
            mode="train_ordered",
            diverse_opponent_actor_count=0,
            diverse_opponent_batch_fraction=0.0,
        )


def test_select_pending_unrolls_async_reserves_diverse_lane_quota() -> None:
    pending = [
        _pending(actor_id=4, unroll_seq=0, behavior_policy_version=1),
        _pending(actor_id=0, unroll_seq=0, behavior_policy_version=1),
        _pending(actor_id=5, unroll_seq=0, behavior_policy_version=1),
        _pending(actor_id=1, unroll_seq=0, behavior_policy_version=1),
    ]

    selected = select_pending_unrolls(
        pending,
        batch_size=3,
        mode="train_async",
        diverse_opponent_actor_count=2,
        diverse_opponent_batch_fraction=0.5,
    )

    assert [item.actor_id for item in selected] == [0, 1, 4]
    assert pending_diverse_unroll_count(selected, diverse_opponent_actor_count=2) == 2


def test_diverse_batch_target_count_clamps_fraction_and_empty_cases() -> None:
    assert (
        diverse_batch_target_count(
            batch_size=4,
            diverse_opponent_actor_count=2,
            diverse_opponent_batch_fraction=0.25,
        )
        == 1
    )
    assert (
        diverse_batch_target_count(
            batch_size=4,
            diverse_opponent_actor_count=2,
            diverse_opponent_batch_fraction=2.0,
        )
        == 4
    )
    assert (
        diverse_batch_target_count(
            batch_size=4,
            diverse_opponent_actor_count=0,
            diverse_opponent_batch_fraction=0.5,
        )
        == 0
    )


class _PendingRuntime(QueueRuntimePendingMixin):
    pass


class _OutcomeTracker:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str]] = []

    def update(self, opponent_id: str, outcome: str) -> None:
        self.updates.append((opponent_id, outcome))


def _process_runtime_with_queue(result_queue: object) -> _PendingRuntime:
    runtime = _PendingRuntime()
    runtime._collector_result_queue = result_queue
    runtime._collector_processes = []
    runtime._pending_unrolls = deque()
    runtime._use_shared_collector_transport = False
    runtime._collector_shared_slots = {}
    runtime._collector_free_queues = []
    runtime._diverse_opponent_batch_wait_ms = 0
    runtime._outcomes = _OutcomeTracker()
    runtime.config = SimpleNamespace(queue_capacity_unrolls=4)
    return runtime


def test_fill_pending_unrolls_raises_child_collector_error_payload() -> None:
    class _ErrorQueue:
        def get(self, *, timeout: float) -> dict[str, object]:
            assert timeout > 0.0
            return {
                "kind": "collector_error_v1",
                "actor_id": 2,
                "error_type": "TypeError",
                "message": "sample() got an unexpected keyword",
                "traceback": "Traceback text",
            }

    runtime = _process_runtime_with_queue(_ErrorQueue())

    with pytest.raises(RuntimeError, match="process collector 2 failed with TypeError"):
        runtime._fill_pending_unrolls(target_count=1, occupancy_samples=[])


def test_fill_pending_unrolls_raises_when_child_collector_exits_without_payload() -> None:
    class _EmptyQueue:
        def get(self, *, timeout: float) -> object:
            raise queue.Empty

    runtime = _process_runtime_with_queue(_EmptyQueue())
    runtime._collector_processes = [SimpleNamespace(pid=1234, exitcode=1)]

    with pytest.raises(RuntimeError, match=r"process collector 0 exited unexpectedly .*exitcode=1"):
        runtime._fill_pending_unrolls(target_count=1, occupancy_samples=[])


def test_fill_pending_unrolls_applies_process_outcome_counter_payloads() -> None:
    class _PayloadQueue:
        def get(self, *, timeout: float) -> dict[str, object]:
            assert timeout > 0.0
            return {
                "actor_id": 0,
                "counters": {
                    "outcome_v1|w|policy_a": 2,
                    "outcome_v1|l|policy_b": 1,
                    "ordinary_counter": 99,
                },
            }

    runtime = _process_runtime_with_queue(_PayloadQueue())

    runtime._fill_pending_unrolls(target_count=1, occupancy_samples=[])

    assert runtime._outcomes.updates == [
        ("policy_a", "w"),
        ("policy_a", "w"),
        ("policy_b", "l"),
    ]
    assert len(runtime._pending_unrolls) == 1
