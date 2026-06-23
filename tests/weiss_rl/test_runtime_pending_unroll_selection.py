from __future__ import annotations

from collections import deque
from typing import Any, cast

from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig

from .runtime_test_support import _make_runtime_unroll


def test_select_pending_unrolls_train_ordered_keeps_same_behavior_version() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=2,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=3,
        queue_capacity_unrolls=3,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=1, behavior_policy_version=1),
            _make_runtime_unroll(actor_id=1, unroll_seq=1, behavior_policy_version=1),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert [(item.behavior_policy_version, item.unroll_seq, item.actor_id) for item in selected] == [
        (0, 0, 0),
        (0, 0, 1),
    ]


def test_select_pending_unrolls_reserves_diverse_lane_quota_in_async_mode() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=6,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._diverse_opponent_actor_count = 2
    runtime_any._diverse_opponent_batch_fraction = 0.5
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=4, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=5, unroll_seq=1, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=2, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=1, unroll_seq=3, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=3, unroll_seq=4, behavior_policy_version=0),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert len(selected) == 4
    assert sum(1 for item in selected if item.actor_id in {0, 1}) == 2
    assert [item.actor_id for item in selected[:2]] == [0, 1]
