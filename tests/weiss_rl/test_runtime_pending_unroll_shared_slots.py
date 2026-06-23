from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from weiss_rl.runtime import (
    QueueRuntime,
)
from weiss_rl.runtime.components.ipc_shared.shared_transport import (
    open_shared_collector_slot,
    shared_unroll_metadata,
    write_unroll_to_shared_slot,
)
from weiss_rl.runtime.components.shared_memory.slots import SharedPendingUnroll

from tests.weiss_rl.runtime_shared_memory_test_support import make_packed_shared_unroll, make_shared_slot_config


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

    monkeypatch.setattr("weiss_rl.runtime.queue_runtime._read_unroll_from_shared_slot", fake_read)

    occupancy_samples: list[float] = []
    runtime._fill_pending_unrolls(target_count=2, occupancy_samples=occupancy_samples)

    assert len(runtime_any._pending_unrolls) == 2
    assert all(not isinstance(item, SharedPendingUnroll) for item in runtime_any._pending_unrolls)
    assert copied_payloads == [(0, 0), (0, 0)]
    assert freed_slots == [0, 0]


def test_shared_pending_unroll_keeps_shared_views_until_release() -> None:
    slot_config = make_shared_slot_config(actor_id=1, slot_id=3)
    slot = open_shared_collector_slot(slot_config, create=True)
    try:
        unroll = make_packed_shared_unroll(
            actor_id=1,
            unroll_seq=9,
            behavior_policy_version=4,
            unroll_hash="shared-view",
            opponent_context_index=np.array([[4, 3], [2, 1]], dtype=np.int16),
            trajectory_retention_valid=np.array([[True, False], [False, True]], dtype=np.bool_),
        )

        write_unroll_to_shared_slot(slot, unroll)
        pending = SharedPendingUnroll.from_metadata(slot, shared_unroll_metadata(unroll, slot_id=3))

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
