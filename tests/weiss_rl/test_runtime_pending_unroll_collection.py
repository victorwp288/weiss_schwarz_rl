from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig, RuntimeUnroll

from tests.weiss_rl.runtime_test_support import _make_runtime_unroll


def test_fill_pending_unrolls_waits_for_diverse_lane_payloads_when_quota_enabled() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    class _ResultQueue:
        def __init__(self, payloads: list[RuntimeUnroll]) -> None:
            self._payloads = deque(payloads)

        def get(self, timeout: float | None = None) -> RuntimeUnroll:
            del timeout
            if not self._payloads:
                raise queue.Empty
            return self._payloads.popleft()

    runtime_any.config = SimpleNamespace(queue_capacity_unrolls=8)
    runtime_any._collector_result_queue = _ResultQueue(
        [
            _make_runtime_unroll(actor_id=4, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=5, unroll_seq=1, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=2, behavior_policy_version=0),
        ]
    )
    runtime_any._use_shared_collector_transport = False
    runtime_any._pending_unrolls = deque()
    runtime_any._outcomes = SimpleNamespace(update=lambda *_args: None)
    runtime_any._diverse_opponent_actor_count = 2
    runtime_any._diverse_opponent_batch_fraction = 0.5
    runtime_any._diverse_opponent_batch_wait_ms = 50

    occupancy_samples: list[float] = []
    QueueRuntime._fill_pending_unrolls(runtime, target_count=2, occupancy_samples=occupancy_samples)

    assert [item.actor_id for item in runtime_any._pending_unrolls] == [4, 5, 0]
    assert QueueRuntime._pending_diverse_unroll_count(runtime) == 1


def test_fill_pending_unrolls_uses_parallel_executor_for_distinct_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=4,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._actors = [cast(Any, SimpleNamespace(actor_id=actor_id)) for actor_id in range(4)]
    runtime_any._collector_result_queue = None
    runtime_any._pending_unrolls = deque()
    runtime_any._next_actor_index = 0
    runtime_any._use_central_batched_collection = False

    submitted_actor_ids: list[int] = []

    class _ImmediateFuture:
        def __init__(self, value: RuntimeUnroll) -> None:
            self._value = value

        def result(self) -> RuntimeUnroll:
            return self._value

    class _FakeExecutor:
        def submit(self, fn, actor):
            submitted_actor_ids.append(int(actor.actor_id))
            return _ImmediateFuture(fn(actor))

    runtime_any._collector_executor = _FakeExecutor()
    runtime_any._collect_actor_unroll = lambda actor: _make_runtime_unroll(
        actor_id=int(actor.actor_id),
        unroll_seq=0,
        behavior_policy_version=0,
    )

    occupancy_samples: list[float] = []
    QueueRuntime._fill_pending_unrolls(runtime, target_count=4, occupancy_samples=occupancy_samples)

    assert submitted_actor_ids == [0, 1, 2, 3]
    assert [item.actor_id for item in runtime_any._pending_unrolls] == [0, 1, 2, 3]
    assert occupancy_samples == [0.0]
