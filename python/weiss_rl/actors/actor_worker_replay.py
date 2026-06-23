"""ActorWorker replay-buffer and numeric-fault adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.actors.actor_faults import write_actor_numeric_fault_bundle
from weiss_rl.actors.replay_capture import (
    append_replay_step,
    clear_replay_for_env,
    ensure_episode_buffers,
    flush_replay_for_env,
    resolve_replay_episode_seed64,
    sync_replay_episode_buffers,
)


def actor_fault_dir_path(worker: Any) -> Path:
    if worker.fault_dir is not None:
        return worker.fault_dir
    if worker.checkpoint_dir is not None:
        return worker.checkpoint_dir / "faults"
    return Path("faults")


def ensure_actor_episode_buffers(worker: Any) -> None:
    worker._episode_buffers_by_env = ensure_episode_buffers(
        worker._episode_buffers_by_env,
        num_envs=worker.num_envs,
    )


def resolve_actor_replay_episode_seed64(worker: Any, episode_seed: np.ndarray | None, *, num_envs: int) -> np.ndarray:
    episode_seed64, worker.episode_seed64_by_env = resolve_replay_episode_seed64(
        episode_seed,
        worker.episode_seed64_by_env,
        seed=worker.seed,
        actor_id=worker.actor_id,
        num_envs=num_envs,
    )
    return episode_seed64


def sync_actor_replay_episode_buffers(
    worker: Any,
    *,
    episode_seed64: np.ndarray,
    simulator_episode_key: np.ndarray | None,
) -> None:
    ensure_actor_episode_buffers(worker)
    if worker.episode_index_by_env is None:
        worker.episode_index_by_env = np.zeros((worker.num_envs,), dtype=np.int64)
    sync_replay_episode_buffers(
        worker._episode_buffers_by_env,
        episode_index_by_env=worker.episode_index_by_env,
        episode_seed64=episode_seed64,
        simulator_episode_key=simulator_episode_key,
        num_envs=worker.num_envs,
    )


def clear_actor_replay_for_env(worker: Any, *, env_index: int) -> None:
    clear_replay_for_env(worker._episode_buffers_by_env, env_index=env_index)


def append_actor_replay_step(
    worker: Any,
    *,
    env_index: int,
    t: int,
    decision_id: int,
    actor: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    engine_status: int,
    legal_ids: np.ndarray,
) -> None:
    ensure_actor_episode_buffers(worker)
    append_replay_step(
        worker._episode_buffers_by_env,
        spec_hash256=worker.spec_hash256,
        env_index=env_index,
        t=t,
        decision_id=decision_id,
        actor=actor,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        engine_status=engine_status,
        legal_ids=legal_ids,
    )


def flush_actor_replay_for_env(worker: Any, *, env_index: int, fault_payload: dict[str, Any] | None = None) -> None:
    if worker.episode_index_by_env is None:
        return
    flush_replay_for_env(
        worker._episode_buffers_by_env,
        env_index=env_index,
        replay_dir=worker.replay_dir,
        checkpoint_dir=worker.checkpoint_dir,
        run_id256=worker.run_id256,
        spec_hash256=worker.spec_hash256,
        actor_id=int(worker.actor_id),
        env_id_base=int(worker.env_id_base),
        replay_rerun_contract=worker.replay_rerun_contract,
        fault_payload=fault_payload,
    )


def raise_actor_numeric_fault(
    worker: Any,
    reason: str,
    *,
    step: int,
    obs: np.ndarray,
    to_play: np.ndarray,
    decision_id: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
    logits: np.ndarray,
    actions: np.ndarray | None = None,
    logp: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    legal_ids: np.ndarray | None = None,
    legal_offsets: np.ndarray | None = None,
    legal_mask: np.ndarray | None = None,
) -> None:
    fault_path, payload = write_actor_numeric_fault_bundle(
        fault_dir=actor_fault_dir_path(worker),
        reason=reason,
        actor_id=worker.actor_id,
        layout_name=worker.layout_name,
        update_count=worker.update_count,
        observed_checkpoint_update=worker.observed_checkpoint_update,
        step=step,
        obs=obs,
        to_play=to_play,
        decision_id=decision_id,
        episode_seed=episode_seed,
        episode_key=episode_key,
        logits=logits,
        actions=actions,
        logp=logp,
        entropy=entropy,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_mask=legal_mask,
    )
    try:
        ensure_actor_episode_buffers(worker)
        for env_index in range(worker.num_envs):
            flush_actor_replay_for_env(worker, env_index=env_index, fault_payload=payload)
    except Exception:
        pass
    raise RuntimeError(f"{reason}; wrote fault bundle to {fault_path}")


__all__ = [
    "actor_fault_dir_path",
    "append_actor_replay_step",
    "clear_actor_replay_for_env",
    "ensure_actor_episode_buffers",
    "flush_actor_replay_for_env",
    "raise_actor_numeric_fault",
    "resolve_actor_replay_episode_seed64",
    "sync_actor_replay_episode_buffers",
]
