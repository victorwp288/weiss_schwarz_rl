"""QueueRuntime pending-unroll scheduling and fill adapters."""

from __future__ import annotations

import queue
import time
from collections.abc import Mapping, Sequence
from typing import Any

from weiss_rl.runtime.components import shared as runtime_shared
from weiss_rl.runtime.components.collection.actor_scheduling import next_actor_batch
from weiss_rl.runtime.components.collection.pending import (
    actor_id_is_diverse_lane,
    diverse_batch_target_count,
    pending_diverse_unroll_count,
    pending_unroll_is_diverse_lane,
    select_pending_unrolls,
)
from weiss_rl.runtime.components.outcomes import apply_outcome_counters_to_tracker
from weiss_rl.runtime.components.types import PendingUnroll, RuntimeUnroll

_SharedPendingUnroll = runtime_shared.SharedPendingUnroll


class QueueRuntimePendingMixin:
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _select_pending_unrolls(self) -> list[PendingUnroll]:
        return select_pending_unrolls(
            tuple(self._pending_unrolls),
            batch_size=int(self.config.batch_unrolls_per_update),
            mode=str(self.config.mode),
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
            diverse_opponent_batch_fraction=float(getattr(self, "_diverse_opponent_batch_fraction", 0.0)),
        )

    def _release_shared_pending_unrolls(self, unrolls: Sequence[PendingUnroll]) -> None:
        if not self._use_shared_collector_transport:
            return
        for unroll in unrolls:
            if not isinstance(unroll, _SharedPendingUnroll):
                continue
            self._collector_free_queues[int(unroll.actor_id)].put(int(unroll.slot_id))

    def _shared_collector_slot_capacity(self) -> int:
        if not self._use_shared_collector_transport:
            return 0
        return int(sum(len(slots) for slots in self._collector_shared_slots.values()))

    def _collector_process_failure_message(self) -> str | None:
        for actor_index, process in enumerate(getattr(self, "_collector_processes", ())):
            exitcode = getattr(process, "exitcode", None)
            if exitcode is None:
                continue
            pid = getattr(process, "pid", None)
            pid_text = "unknown" if pid is None else str(pid)
            return f"process collector {actor_index} exited unexpectedly (pid={pid_text}, exitcode={int(exitcode)})"
        return None

    def _raise_if_collector_process_failed(self) -> None:
        message = self._collector_process_failure_message()
        if message is not None:
            raise RuntimeError(message)

    def _raise_if_collector_error_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        if str(payload.get("kind", "")).strip() != "collector_error_v1":
            return
        actor_id = int(payload.get("actor_id", -1))
        error_type = str(payload.get("error_type", "Exception")).strip() or "Exception"
        message = str(payload.get("message", "")).strip()
        traceback_text = str(payload.get("traceback", "")).strip()
        details = f": {message}" if message else ""
        if traceback_text:
            details = f"{details}\n{traceback_text}"
        raise RuntimeError(f"process collector {actor_id} failed with {error_type}{details}")

    def _apply_collector_outcome_counters(self, payload: Any) -> None:
        counters: dict[str, int] | None = None
        if isinstance(payload, RuntimeUnroll):
            raw_counters = getattr(payload, "counters", None)
            if isinstance(raw_counters, dict):
                counters = {str(key): int(value) for key, value in raw_counters.items()}
        elif isinstance(payload, Mapping):
            raw_counters = payload.get("counters")
            if isinstance(raw_counters, Mapping):
                counters = {str(key): int(value) for key, value in raw_counters.items()}
        apply_outcome_counters_to_tracker(outcome_tracker=self._outcomes, counters=counters)

    def _next_actor_batch(self, count: int) -> list[Any]:
        actor_batch, next_actor_index = next_actor_batch(
            self._actors,
            next_actor_index=int(getattr(self, "_next_actor_index", 0)),
            count=int(count),
        )
        self._next_actor_index = next_actor_index
        return actor_batch

    def _fill_pending_unrolls(self, *, target_count: int, occupancy_samples: list[float]) -> None:
        if self._collector_result_queue is not None:
            eager_spill_shared_slots = self._use_shared_collector_transport and int(target_count) > max(
                1, self._shared_collector_slot_capacity()
            )
            diverse_target = self._diverse_batch_target_count(int(target_count))
            wait_deadline: float | None = None
            while True:
                if len(self._pending_unrolls) >= int(target_count):
                    if diverse_target <= 0 or self._pending_diverse_unroll_count() >= diverse_target:
                        break
                    if wait_deadline is None:
                        wait_deadline = time.perf_counter() + (float(self._diverse_opponent_batch_wait_ms) / 1000.0)
                    remaining_wait = wait_deadline - time.perf_counter()
                    if remaining_wait <= 0.0:
                        break
                    queue_timeout: float | None = remaining_wait
                else:
                    queue_timeout = None
                occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
                try:
                    payload = self._collector_result_queue.get(
                        timeout=0.1 if queue_timeout is None else min(float(queue_timeout), 0.1)
                    )
                except queue.Empty:
                    self._raise_if_collector_process_failed()
                    continue
                self._raise_if_collector_error_payload(payload)
                self._apply_collector_outcome_counters(payload)
                if (not self._use_shared_collector_transport) or isinstance(payload, RuntimeUnroll):
                    self._pending_unrolls.append(payload)
                    continue
                actor_id = int(payload["actor_id"])
                slot_id = int(payload.get("slot_id", 0))
                slot = self._collector_shared_slots[actor_id][slot_id]
                if eager_spill_shared_slots:
                    self._pending_unrolls.append(self._read_unroll_from_shared_slot(slot, payload))
                    self._collector_free_queues[actor_id].put(slot_id)
                else:
                    self._pending_unrolls.append(_SharedPendingUnroll.from_metadata(slot, payload))
            return
        if self._use_central_batched_collection:
            while len(self._pending_unrolls) < int(target_count):
                occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
                remaining = int(target_count) - len(self._pending_unrolls)
                actors = self._next_actor_batch(remaining)
                if not actors:
                    break
                self._pending_unrolls.extend(self._collect_actor_unrolls_central(actors))
            return
        while len(self._pending_unrolls) < int(target_count):
            occupancy_samples.append(len(self._pending_unrolls) / float(self.config.queue_capacity_unrolls))
            remaining = int(target_count) - len(self._pending_unrolls)
            actors = self._next_actor_batch(remaining)
            if not actors:
                break
            if self._collector_executor is None or len(actors) == 1:
                for actor in actors:
                    self._pending_unrolls.append(self._collect_actor_unroll(actor))
                continue
            futures = [self._collector_executor.submit(self._collect_actor_unroll, actor) for actor in actors]
            for future in futures:
                self._pending_unrolls.append(future.result())

    def _actor_id_is_diverse_lane(self, actor_id: int) -> bool:
        return actor_id_is_diverse_lane(
            actor_id=int(actor_id),
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
        )

    def _pending_unroll_is_diverse_lane(self, item: PendingUnroll) -> bool:
        return pending_unroll_is_diverse_lane(
            item,
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
        )

    def _pending_diverse_unroll_count(self) -> int:
        return pending_diverse_unroll_count(
            tuple(self._pending_unrolls),
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
        )

    def _diverse_batch_target_count(self, batch_size: int) -> int:
        return diverse_batch_target_count(
            batch_size=int(batch_size),
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
            diverse_opponent_batch_fraction=float(getattr(self, "_diverse_opponent_batch_fraction", 0.0)),
        )
