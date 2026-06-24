"""Actor episode-boundary accounting, replay flushing, and autoreset handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.actors.actor_worker_helpers import update_outcomes
from weiss_rl.core.termination_reason import classify_episode_end_reason
from weiss_rl.diagnostics.probes.action_diagnostics import reset_action_sequence_state


@dataclass(slots=True)
class ActorEpisodeCounters:
    engine_fault_done_rows: int = 0
    no_progress_timeout_rows: int = 0
    natural_timeout_rows: int = 0
    decision_limit_timeout_rows: int = 0
    tick_limit_timeout_rows: int = 0
    timeout_unknown_rows: int = 0

    def record(self, termination_reason: str) -> None:
        if termination_reason == "engine_fault":
            self.engine_fault_done_rows += 1
        elif termination_reason == "no_progress_timeout":
            self.no_progress_timeout_rows += 1
        elif termination_reason == "decision_limit_timeout":
            self.natural_timeout_rows += 1
            self.decision_limit_timeout_rows += 1
        elif termination_reason == "tick_limit_timeout":
            self.natural_timeout_rows += 1
            self.tick_limit_timeout_rows += 1
        elif termination_reason == "timeout_unknown":
            self.natural_timeout_rows += 1
            self.timeout_unknown_rows += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "engine_fault_done_rows": self.engine_fault_done_rows,
            "no_progress_timeout_rows": self.no_progress_timeout_rows,
            "natural_timeout_rows": self.natural_timeout_rows,
            "decision_limit_timeout_rows": self.decision_limit_timeout_rows,
            "tick_limit_timeout_rows": self.tick_limit_timeout_rows,
            "timeout_unknown_rows": self.timeout_unknown_rows,
        }


def handle_actor_episode_boundaries(
    worker: Any,
    *,
    env: Any,
    next_batch: Any,
    done_mask: np.ndarray,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    engine_status: np.ndarray,
    decision_count: np.ndarray,
    tick_count: np.ndarray,
    no_progress_count: np.ndarray,
    timeout_limits: dict[str, int | None],
    action_accounting: Any,
    episode_counters: ActorEpisodeCounters,
    t: int,
    decision_id: np.ndarray,
    replay_episode_seed64: np.ndarray,
    simulator_episode_key: np.ndarray | None,
) -> Any:
    update_outcomes(
        worker.outcomes,
        opponent_ids=worker.opponent_id_by_env,
        reward=reward,
        engine_status=engine_status,
        done=done_mask,
    )

    for env_index in np.flatnonzero(done_mask):
        env_index_int = int(env_index)
        termination_reason = classify_episode_end_reason(
            terminated=bool(terminated[env_index_int]),
            truncated=bool(truncated[env_index_int]),
            engine_status=int(engine_status[env_index_int]),
            decision_count=int(decision_count[env_index_int]),
            tick_count=int(tick_count[env_index_int]),
            no_progress_count=int(no_progress_count[env_index_int]),
            max_decisions=timeout_limits["max_decisions"],
            max_ticks=timeout_limits["max_ticks"],
            max_no_progress_decisions=timeout_limits["max_no_progress_decisions"],
        )
        episode_counters.record(termination_reason)
        fault_payload = _engine_fault_payload(
            worker,
            t=t,
            env_index=env_index_int,
            decision_id=decision_id,
            engine_status=engine_status,
            terminated=terminated,
            truncated=truncated,
            replay_episode_seed64=replay_episode_seed64,
            simulator_episode_key=simulator_episode_key,
        )
        if worker.capture_replays_on_done or fault_payload is not None:
            worker._flush_replay_for_env(env_index=env_index_int, fault_payload=fault_payload)
        else:
            worker._clear_replay_for_env(env_index=env_index_int)

    if worker.episode_index_by_env is not None:
        worker.episode_index_by_env[done_mask] += 1
    if worker.episode_seed64_by_env is not None:
        worker.episode_seed64_by_env[done_mask] += np.uint64(1)

    reset_action_sequence_state(action_accounting.sequence_state, done_mask)
    reset_done = getattr(env, "reset_done", None)
    if callable(reset_done):
        worker._resample_opponents(done_mask)
        return reset_done(done_mask)
    return next_batch


def _engine_fault_payload(
    worker: Any,
    *,
    t: int,
    env_index: int,
    decision_id: np.ndarray,
    engine_status: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    replay_episode_seed64: np.ndarray,
    simulator_episode_key: np.ndarray | None,
) -> dict[str, Any] | None:
    if int(engine_status[env_index]) == 0:
        return None
    return {
        "format": "engine_fault_replay",
        "actor_id": worker.actor_id,
        "env_id": int(worker.env_id_base + env_index),
        "t": int(t),
        "decision_id": int(decision_id[env_index]),
        "engine_status": int(engine_status[env_index]),
        "terminated": bool(terminated[env_index]),
        "truncated": bool(truncated[env_index]),
        "episode_seed64": int(replay_episode_seed64[env_index]),
        "simulator_episode_key": None if simulator_episode_key is None else int(simulator_episode_key[env_index]),
    }


__all__ = ["ActorEpisodeCounters", "handle_actor_episode_boundaries"]
