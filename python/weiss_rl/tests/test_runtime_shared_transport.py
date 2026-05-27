from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import (
    QueueRuntime,
    QueueRuntimeConfig,
    RuntimeUnroll,
    _create_shared_collector_slot_config,
    _open_shared_collector_slot,
    _read_unroll_from_shared_slot,
    _shared_unroll_metadata,
    _SharedPendingUnroll,
    _write_unroll_to_shared_slot,
)


def _make_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=f"{actor_id}:{unroll_seq}:{behavior_policy_version}",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        behavior_logits=None,
        counters=counters,
    )


def test_fill_pending_unrolls_spills_shared_slots_when_target_exceeds_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)

    class _ResultQueue:
        def __init__(self, payloads: list[dict[str, Any]]) -> None:
            self._payloads = deque(payloads)

        def get(self, timeout: float | None = None) -> dict[str, Any]:
            del timeout
            return self._payloads.popleft()

    freed_slots: list[int] = []
    copied_payloads: list[tuple[int, int]] = []

    runtime_any._collector_result_queue = _ResultQueue(
        [
            {"actor_id": 0, "slot_id": 0, "unroll_seq": 1, "behavior_policy_version": 1},
            {"actor_id": 0, "slot_id": 0, "unroll_seq": 2, "behavior_policy_version": 1},
        ]
    )
    runtime_any._use_shared_collector_transport = True
    runtime_any._collector_shared_slots = {0: (object(),)}
    runtime_any._collector_free_queues = [SimpleNamespace(put=lambda slot_id: freed_slots.append(int(slot_id)))]
    runtime_any._pending_unrolls = deque()
    runtime_any._outcomes = SimpleNamespace(update=lambda *_args: None)
    runtime_any.config = SimpleNamespace(queue_capacity_unrolls=8)

    def fake_read(slot: object, metadata: dict[str, Any]) -> SimpleNamespace:
        del slot
        copied_payloads.append((int(metadata["actor_id"]), int(metadata["slot_id"])))
        return SimpleNamespace(kind="copied", unroll_seq=int(metadata["unroll_seq"]))

    monkeypatch.setattr("weiss_rl.runtime._read_unroll_from_shared_slot", fake_read)

    occupancy_samples: list[float] = []
    runtime._fill_pending_unrolls(target_count=2, occupancy_samples=occupancy_samples)

    assert len(runtime_any._pending_unrolls) == 2
    assert all(not isinstance(item, _SharedPendingUnroll) for item in runtime_any._pending_unrolls)
    assert copied_payloads == [(0, 0), (0, 0)]
    assert freed_slots == [0, 0]


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


def test_shared_collector_slot_round_trip_preserves_packed_unroll_payload() -> None:
    slot_config = _create_shared_collector_slot_config(
        actor_id=0,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )
    slot = _open_shared_collector_slot(slot_config, create=True)
    try:
        packed = LegalActionBatch.from_packed(
            np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
            np.array([0, 2, 3, 5, 6], dtype=np.uint32),
            meta=np.array(
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [2, 0, 0, 0],
                    [3, 0, 1, 0],
                    [3, 1, 1, 0],
                    [8, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            action_space=5,
        )
        unroll = RuntimeUnroll(
            actor_id=0,
            unroll_seq=7,
            behavior_policy_version=11,
            unroll_hash="roundtrip",
            obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
            truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
            to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
            behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            legal_actions=packed,
            bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
            bootstrap_actor=np.array([0, 1], dtype=np.int64),
            bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
            initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
            episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
            policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
            opponent_context_index=np.array([[1, 2], [3, 4]], dtype=np.int16),
            teacher_family=np.array([[1, 2], [3, -1]], dtype=np.int32),
            teacher_slot=np.array([[0, -1], [2, -1]], dtype=np.int32),
            teacher_move_source=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_attack_type=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_action=np.array([[4, 9], [12, -1]], dtype=np.int32),
            teacher_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
            trajectory_retention_valid=np.array([[False, True], [True, False]], dtype=np.bool_),
            behavior_logits=None,
        )

        _write_unroll_to_shared_slot(slot, unroll)
        metadata = _shared_unroll_metadata(unroll)
        restored = _read_unroll_from_shared_slot(slot, metadata)

        assert metadata["has_trajectory_retention_label"] is True
        assert metadata["has_opponent_context_index"] is True
        assert restored.actor_id == unroll.actor_id
        assert restored.unroll_seq == unroll.unroll_seq
        assert restored.behavior_policy_version == unroll.behavior_policy_version
        assert np.array_equal(restored.obs, unroll.obs)
        assert np.array_equal(restored.actions, unroll.actions)
        assert np.array_equal(restored.bootstrap_obs, unroll.bootstrap_obs)
        assert np.array_equal(restored.bootstrap_value, unroll.bootstrap_value)
        assert np.array_equal(restored.final_hidden_state, unroll.final_hidden_state)
        assert np.array_equal(
            cast(np.ndarray, restored.opponent_context_index),
            cast(np.ndarray, unroll.opponent_context_index),
        )
        assert np.array_equal(cast(np.ndarray, restored.teacher_family), cast(np.ndarray, unroll.teacher_family))
        assert np.array_equal(cast(np.ndarray, restored.teacher_slot), cast(np.ndarray, unroll.teacher_slot))
        assert np.array_equal(
            cast(np.ndarray, restored.teacher_move_source),
            cast(np.ndarray, unroll.teacher_move_source),
        )
        assert np.array_equal(
            cast(np.ndarray, restored.teacher_attack_type),
            cast(np.ndarray, unroll.teacher_attack_type),
        )
        assert np.array_equal(cast(np.ndarray, restored.teacher_action), cast(np.ndarray, unroll.teacher_action))
        assert np.array_equal(cast(np.ndarray, restored.teacher_valid), cast(np.ndarray, unroll.teacher_valid))
        assert np.array_equal(
            cast(np.ndarray, restored.trajectory_retention_valid),
            cast(np.ndarray, unroll.trajectory_retention_valid),
        )
        assert restored.legal_actions.ids is not None
        assert restored.legal_actions.offsets is not None
        assert restored.legal_actions.action_space == 5
        assert restored.legal_actions.ids.tolist() == cast(np.ndarray, unroll.legal_actions.ids).tolist()
        assert restored.legal_actions.offsets.tolist() == cast(np.ndarray, unroll.legal_actions.offsets).tolist()
        assert restored.legal_actions.meta is not None
        assert restored.legal_actions.meta.tolist() == cast(np.ndarray, unroll.legal_actions.meta).tolist()
    finally:
        slot.close(unlink=True)


def test_shared_pending_unroll_keeps_shared_views_until_release() -> None:
    slot_config = _create_shared_collector_slot_config(
        actor_id=1,
        slot_id=3,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )
    slot = _open_shared_collector_slot(slot_config, create=True)
    try:
        packed = LegalActionBatch.from_packed(
            np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
            np.array([0, 2, 3, 5, 6], dtype=np.uint32),
            meta=np.array(
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [2, 0, 0, 0],
                    [3, 0, 1, 0],
                    [3, 1, 1, 0],
                    [8, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            action_space=5,
        )
        unroll = RuntimeUnroll(
            actor_id=1,
            unroll_seq=9,
            behavior_policy_version=4,
            unroll_hash="shared-view",
            obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
            truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
            to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
            behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            legal_actions=packed,
            bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
            bootstrap_actor=np.array([0, 1], dtype=np.int64),
            bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
            initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
            episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
            policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
            opponent_context_index=np.array([[4, 3], [2, 1]], dtype=np.int16),
            teacher_family=np.array([[1, 2], [3, -1]], dtype=np.int32),
            teacher_slot=np.array([[0, -1], [2, -1]], dtype=np.int32),
            teacher_move_source=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_attack_type=np.array([[-1, 1], [0, -1]], dtype=np.int32),
            teacher_action=np.array([[4, 9], [12, -1]], dtype=np.int32),
            teacher_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
            trajectory_retention_valid=np.array([[True, False], [False, True]], dtype=np.bool_),
            behavior_logits=None,
        )

        _write_unroll_to_shared_slot(slot, unroll)
        pending = _SharedPendingUnroll.from_metadata(slot, _shared_unroll_metadata(unroll, slot_id=3))

        assert pending.slot_id == 3
        assert pending.obs is slot.obs
        assert np.shares_memory(pending.legal_actions.ids, slot.legal_ids)
        assert np.shares_memory(pending.legal_actions.meta, slot.legal_action_meta)
        assert pending.teacher_move_source is slot.teacher_move_source
        assert pending.teacher_action is slot.teacher_action
        assert pending.trajectory_retention_valid is slot.trajectory_retention_valid
        assert pending.opponent_context_index is slot.opponent_context_index

        runtime = object.__new__(QueueRuntime)
        runtime_any = cast(Any, runtime)
        runtime_any._use_shared_collector_transport = True
        runtime_any._collector_free_queues = [queue.Queue(), queue.Queue()]

        QueueRuntime._release_shared_pending_unrolls(runtime, [pending])

        assert runtime_any._collector_free_queues[1].get_nowait() == 3
    finally:
        slot.close(unlink=True)
