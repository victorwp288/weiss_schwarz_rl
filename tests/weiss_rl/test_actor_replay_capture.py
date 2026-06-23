from __future__ import annotations

from pathlib import Path

import numpy as np
from weiss_rl.actors.replay_capture import (
    ReplayEpisodeBuffer,
    actor_replay_dir,
    append_replay_step,
    ensure_episode_buffers,
    resolve_replay_episode_seed64,
    sync_replay_episode_buffers,
)


def test_actor_replay_dir_defaults_near_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "runs" / "demo" / "training" / "checkpoints"

    assert (
        actor_replay_dir(replay_dir=None, checkpoint_dir=checkpoint_dir)
        == tmp_path / "runs" / "demo" / "replays" / "regression"
    )


def test_resolve_replay_episode_seed64_reuses_actor_local_seed_state() -> None:
    seed64, state = resolve_replay_episode_seed64(
        None,
        None,
        seed=3,
        actor_id=2,
        num_envs=2,
    )

    assert state is seed64
    assert seed64.dtype == np.uint64
    assert seed64.tolist() == [8589934595, 8589934596]


def test_sync_replay_episode_buffers_tracks_episode_identity_changes() -> None:
    buffers = ensure_episode_buffers([], num_envs=1)
    episode_index_by_env = np.array([4], dtype=np.int64)

    sync_replay_episode_buffers(
        buffers,
        episode_index_by_env=episode_index_by_env,
        episode_seed64=np.array([10], dtype=np.uint64),
        simulator_episode_key=np.array([111], dtype=np.uint64),
        num_envs=1,
    )
    first = buffers[0]
    assert isinstance(first, ReplayEpisodeBuffer)
    assert first.actor_episode_index == 4
    assert first.episode_seed64 == 10
    assert first.simulator_episode_key == 111

    sync_replay_episode_buffers(
        buffers,
        episode_index_by_env=episode_index_by_env,
        episode_seed64=np.array([10], dtype=np.uint64),
        simulator_episode_key=np.array([111], dtype=np.uint64),
        num_envs=1,
    )
    assert buffers[0] is first


def test_append_replay_step_uses_pre_step_legal_fingerprint() -> None:
    buffers = [ReplayEpisodeBuffer(actor_episode_index=0, episode_seed64=1, simulator_episode_key=None)]

    append_replay_step(
        buffers,
        spec_hash256=bytes.fromhex("ab" * 32),
        env_index=0,
        t=3,
        decision_id=7,
        actor=1,
        action=2,
        reward=1.5,
        terminated=True,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([1, 2, 3], dtype=np.uint16),
    )

    assert len(buffers[0].steps) == 1
    step = buffers[0].steps[0]
    assert step.t == 3
    assert step.decision_id == 7
    assert step.legal_fingerprint64 != 0
