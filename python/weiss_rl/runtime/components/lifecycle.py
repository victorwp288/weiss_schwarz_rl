"""Runtime lifecycle and snapshot publication helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import queue
import time
from contextlib import suppress
from typing import Any

from weiss_rl.model import PolicyValueModel
from weiss_rl.runtime.components.hashing import hash_state_dict as _hash_state_dict
from weiss_rl.runtime.components.ipc_shared.ipc import serialize_state_dict_for_ipc as _serialize_state_dict_for_ipc


class QueueRuntimeLifecycleMixin:
    def close(self: Any) -> None:
        if self._collector_result_queue is not None:
            for control_queue in self._collector_control_queues:
                control_queue.put({"kind": "stop"})
            if self._use_shared_collector_transport:
                for free_queue in self._collector_free_queues:
                    with suppress(queue.Full):
                        free_queue.put_nowait("stop")
            alive_processes: list[Any] = []
            for process in self._collector_processes:
                process.join(timeout=0.25)
                if process.is_alive():
                    alive_processes.append(process)
            for process in alive_processes:
                process.terminate()
            for process in alive_processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
                        process.join(timeout=0.5)
            for pending_queue in [*self._collector_control_queues, *self._collector_free_queues]:
                cancel_join = getattr(pending_queue, "cancel_join_thread", None)
                if callable(cancel_join):
                    cancel_join()
                close_queue = getattr(pending_queue, "close", None)
                if callable(close_queue):
                    close_queue()
            self._collector_control_queues.clear()
            self._collector_free_queues.clear()
            self._collector_processes.clear()
            cancel_join = getattr(self._collector_result_queue, "cancel_join_thread", None)
            if callable(cancel_join):
                cancel_join()
            self._collector_result_queue.close()
            self._collector_result_queue = None
        if self._use_shared_collector_transport:
            for slots in self._collector_shared_slots.values():
                for slot in slots:
                    slot.close(unlink=True)
            self._collector_shared_slots.clear()
        if self._collector_executor is not None:
            self._collector_executor.shutdown(wait=True)
        for actor in self._actors:
            actor.env.close()

    def maybe_publish_snapshot(
        self: Any,
        *,
        learner_model: PolicyValueModel,
        learner_update_count: int,
        force: bool = False,
    ) -> dict[str, float]:
        self._current_learner_update = int(learner_update_count)
        if self._collector_result_queue is not None and learner_update_count > 0:
            for control_queue in self._collector_control_queues:
                control_queue.put({"kind": "set_update", "update": int(learner_update_count)})
        if learner_update_count <= 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if not force and learner_update_count == self._last_published_snapshot_version:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}
        if not force and learner_update_count % int(self.config.actor_reload_interval_updates) != 0:
            return {"snapshot_publish_latency_ms": 0.0, "snapshot_apply_latency_ms": 0.0}

        publish_started = time.perf_counter()
        state_dict = {key: value.detach().cpu().clone() for key, value in learner_model.state_dict().items()}
        state_fingerprint = _hash_state_dict(state_dict)
        published_snapshot_update_by_fingerprint = getattr(self, "_published_snapshot_update_by_fingerprint", None)
        if published_snapshot_update_by_fingerprint is None:
            published_snapshot_update_by_fingerprint = {}
            self._published_snapshot_update_by_fingerprint = published_snapshot_update_by_fingerprint
        published_snapshot_update = int(
            published_snapshot_update_by_fingerprint.setdefault(state_fingerprint, int(learner_update_count))
        )
        self._effective_learner_update = published_snapshot_update
        publish_latency_ms = (time.perf_counter() - publish_started) * 1000.0

        apply_started = time.perf_counter()
        if self._collector_result_queue is not None:
            serialized_state_dict = _serialize_state_dict_for_ipc(state_dict)
            for control_queue in self._collector_control_queues:
                control_queue.put(
                    {
                        "kind": "reload",
                        "model_state_dict": serialized_state_dict,
                        "update": int(learner_update_count),
                        "effective_update": int(published_snapshot_update),
                    }
                )
        else:
            if self._shared_actor_model is not None:
                self._shared_actor_model.load_state_dict(state_dict)
                self._shared_actor_model.eval()
                for actor in self._actors:
                    actor.snapshot_version = int(learner_update_count)
            else:
                for actor in self._actors:
                    actor.model.load_state_dict(state_dict)
                    actor.model.eval()
                    actor.snapshot_version = int(learner_update_count)
        if self._bootstrap_models is not None:
            for bootstrap_model in self._bootstrap_models:
                bootstrap_model.load_state_dict(state_dict)
                bootstrap_model.eval()
        apply_latency_ms = (time.perf_counter() - apply_started) * 1000.0
        self._last_published_snapshot_version = int(learner_update_count)
        return {
            "snapshot_publish_latency_ms": publish_latency_ms,
            "snapshot_apply_latency_ms": apply_latency_ms,
        }

    def reset_outcome_tracker(self: Any) -> None:
        self._pfsp_epoch = int(self._outcomes.bump_epoch(drop_previous=True))
        self._pfsp_quarantined_opponents = 0

    def _league_reference_update(self: Any) -> int:
        effective_update = int(getattr(self, "_effective_learner_update", 0))
        if effective_update > 0:
            return effective_update
        return int(getattr(self, "_current_learner_update", 0))
