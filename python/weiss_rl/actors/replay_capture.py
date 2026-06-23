"""Replay capture helpers for actor-worker rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.bundles import (
    ReplayRerunContract,
    ReplayStep,
    compute_legal_fingerprint64,
    make_replay_bundle_meta,
    write_replay_bundle,
)


@dataclass(slots=True)
class ReplayEpisodeBuffer:
    actor_episode_index: int
    episode_seed64: int
    simulator_episode_key: int | bytes | None
    steps: list[ReplayStep] = field(default_factory=list)


def actor_replay_dir(*, replay_dir: Path | None, checkpoint_dir: Path | None) -> Path:
    if replay_dir is not None:
        return replay_dir
    if checkpoint_dir is not None:
        return checkpoint_dir.parent.parent / "replays" / "regression"
    return Path("replays") / "regression"


def ensure_episode_buffers(
    buffers_by_env: list[ReplayEpisodeBuffer | None],
    *,
    num_envs: int,
) -> list[ReplayEpisodeBuffer | None]:
    if buffers_by_env:
        return buffers_by_env
    return [None for _ in range(num_envs)]


def resolve_replay_episode_seed64(
    episode_seed: np.ndarray | None,
    episode_seed64_by_env: np.ndarray | None,
    *,
    seed: int,
    actor_id: int,
    num_envs: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if episode_seed is not None:
        return episode_seed, episode_seed64_by_env
    if episode_seed64_by_env is None:
        base_seed64 = np.uint64(seed) ^ (np.uint64(actor_id) << np.uint64(32))
        episode_seed64_by_env = (base_seed64 + np.arange(num_envs, dtype=np.uint64)).astype(np.uint64, copy=False)
    return episode_seed64_by_env, episode_seed64_by_env


def sync_replay_episode_buffers(
    buffers_by_env: list[ReplayEpisodeBuffer | None],
    *,
    episode_index_by_env: np.ndarray,
    episode_seed64: np.ndarray,
    simulator_episode_key: np.ndarray | None,
    num_envs: int,
) -> None:
    for env_index in range(num_envs):
        next_seed = int(episode_seed64[env_index])
        next_key = None if simulator_episode_key is None else int(simulator_episode_key[env_index])
        current = buffers_by_env[env_index]
        if current is not None:
            same_seed = int(current.episode_seed64) == next_seed
            same_key = current.simulator_episode_key == next_key
            if same_seed and same_key:
                continue
        buffers_by_env[env_index] = ReplayEpisodeBuffer(
            actor_episode_index=int(episode_index_by_env[env_index]),
            episode_seed64=next_seed,
            simulator_episode_key=next_key,
        )


def clear_replay_for_env(
    buffers_by_env: list[ReplayEpisodeBuffer | None],
    *,
    env_index: int,
) -> None:
    if not buffers_by_env:
        return
    buffers_by_env[env_index] = None


def append_replay_step(
    buffers_by_env: list[ReplayEpisodeBuffer | None],
    *,
    spec_hash256: bytes | None,
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
    if spec_hash256 is None:
        return
    buffer = buffers_by_env[env_index]
    if buffer is None:
        return
    legal_fingerprint64 = compute_legal_fingerprint64(
        spec_hash256=spec_hash256,
        decision_id=int(decision_id),
        legal_ids=legal_ids,
    )
    buffer.steps.append(
        ReplayStep(
            t=int(t),
            decision_id=int(decision_id),
            actor=int(actor),
            action=int(action),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            engine_status=int(engine_status),
            legal_fingerprint64=int(legal_fingerprint64),
        )
    )


def flush_replay_for_env(
    buffers_by_env: list[ReplayEpisodeBuffer | None],
    *,
    env_index: int,
    replay_dir: Path | None,
    checkpoint_dir: Path | None,
    run_id256: bytes | None,
    spec_hash256: bytes | None,
    actor_id: int,
    env_id_base: int,
    replay_rerun_contract: ReplayRerunContract | None,
    fault_payload: dict[str, Any] | None = None,
) -> None:
    if run_id256 is None or spec_hash256 is None:
        return
    if not buffers_by_env:
        return

    buffer = buffers_by_env[env_index]
    if buffer is None or not buffer.steps:
        clear_replay_for_env(buffers_by_env, env_index=env_index)
        return

    meta = make_replay_bundle_meta(
        simulator_episode_key=buffer.simulator_episode_key,
        run_id256=run_id256,
        spec_hash256=spec_hash256,
        actor_id=int(actor_id),
        env_id=int(env_id_base + env_index),
        episode_index=int(buffer.actor_episode_index),
        episode_seed64=int(buffer.episode_seed64),
        rerun_contract=replay_rerun_contract,
    )
    write_replay_bundle(
        out_dir=actor_replay_dir(replay_dir=replay_dir, checkpoint_dir=checkpoint_dir),
        meta=meta,
        steps=buffer.steps,
        fault_payload=fault_payload,
    )
    clear_replay_for_env(buffers_by_env, env_index=env_index)


__all__ = [
    "ReplayEpisodeBuffer",
    "actor_replay_dir",
    "append_replay_step",
    "clear_replay_for_env",
    "ensure_episode_buffers",
    "flush_replay_for_env",
    "resolve_replay_episode_seed64",
    "sync_replay_episode_buffers",
]
