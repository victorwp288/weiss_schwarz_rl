from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from weiss_rl.actors.actor_worker_replay import actor_fault_dir_path, resolve_actor_replay_episode_seed64


def test_actor_fault_dir_path_prefers_explicit_then_checkpoint() -> None:
    assert actor_fault_dir_path(SimpleNamespace(fault_dir=Path("explicit"), checkpoint_dir=Path("ckpt"))) == Path(
        "explicit"
    )
    assert actor_fault_dir_path(SimpleNamespace(fault_dir=None, checkpoint_dir=Path("ckpt"))) == Path("ckpt/faults")
    assert actor_fault_dir_path(SimpleNamespace(fault_dir=None, checkpoint_dir=None)) == Path("faults")


def test_resolve_actor_replay_episode_seed64_updates_actor_local_seed_state() -> None:
    worker = SimpleNamespace(seed=7, actor_id=2, episode_seed64_by_env=None)

    first = resolve_actor_replay_episode_seed64(worker, None, num_envs=2)
    second = resolve_actor_replay_episode_seed64(worker, None, num_envs=2)

    assert first.dtype == np.uint64
    assert first.tolist() == second.tolist()
    assert worker.episode_seed64_by_env.tolist() == first.tolist()


def test_resolve_actor_replay_episode_seed64_uses_simulator_seed_without_replacing_state() -> None:
    actor_seed_state = np.asarray([11, 12], dtype=np.uint64)
    simulator_seed = np.asarray([101, 102], dtype=np.uint64)
    worker = SimpleNamespace(seed=7, actor_id=2, episode_seed64_by_env=actor_seed_state)

    resolved = resolve_actor_replay_episode_seed64(worker, simulator_seed, num_envs=2)

    assert resolved is simulator_seed
    assert worker.episode_seed64_by_env is actor_seed_state
