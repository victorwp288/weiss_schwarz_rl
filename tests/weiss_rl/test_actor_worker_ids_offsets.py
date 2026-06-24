from __future__ import annotations

import numpy as np
from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.core.masking import masked_logp_from_legal_ids, resolve_pass_action_id
from weiss_rl.diagnostics.probes.action_diagnostics import MAIN_MOVE_BASE

from .actor_worker_test_support import (
    ACTION_SPACE,
    OBS_LEN,
    FakeIdsEnv,
    MainMoveIdWithFalseFlagEnv,
    PaddedIdsEnv,
    ReusingIdsEnv,
    _expected_obs,
    _make_real_env,
    _policy_logits,
    _pool_episode_identity,
    _step_legal_slice,
    _uniform_policy_logits,
)


def test_actor_worker_ids_offsets_preserves_behavior_logp_contract() -> None:
    worker = ActorWorker(
        actor_id=0,
        unroll_length=4,
        num_envs=3,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=7,
    )

    batch = worker.run_once(env=FakeIdsEnv(3, seed=11), policy_logits_fn=_policy_logits)

    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_offsets.shape == (batch.T * batch.N + 1,)

    env_ids_i32 = np.arange(batch.N, dtype=np.int32)
    env_ids_u64 = np.arange(batch.N, dtype=np.uint64)
    pass_action_id = resolve_pass_action_id()

    for t in range(batch.T):
        assert np.array_equal(batch.obs[t], _expected_obs(t, num_envs=batch.N))
        assert np.array_equal(batch.to_play_seat[t], ((env_ids_i32 + t) % 2).astype(np.int8))
        assert np.array_equal(batch.decision_id[t], np.full((batch.N,), t, dtype=np.int32))
        assert np.array_equal(batch.reward[t], np.float32(t + 1) + env_ids_i32.astype(np.float32) * np.float32(0.1))
        assert np.array_equal(batch.terminated[t], (t + 1 + env_ids_i32) % 2 == 1)
        assert np.array_equal(batch.truncated[t], (t + 1 + env_ids_i32) % 3 == 2)
        assert np.array_equal(batch.engine_status[t], 100 + (t + 1) * 10 + env_ids_i32)
        assert np.array_equal(batch.episode_seed[t], np.uint64(10_000 + t * 10) + env_ids_u64)
        assert np.array_equal(batch.episode_key[t], np.uint64(20_000 + t * 10) + env_ids_u64)

        logits = _policy_logits(batch.obs[t], batch.to_play_seat[t])
        legal_ids, legal_offsets = _step_legal_slice(batch, t)
        recomputed = masked_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            batch.action[t].astype(np.int64, copy=False),
            pass_action_id=pass_action_id,
        )
        assert np.array_equal(recomputed, batch.behavior_logp[t])


def test_actor_worker_ids_offsets_discards_raw_capacity_tail_between_steps() -> None:
    worker = ActorWorker(
        actor_id=9,
        unroll_length=3,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=23,
    )

    env = PaddedIdsEnv(2, seed=29)
    batch = worker.run_once(env=env, policy_logits_fn=_policy_logits)

    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_ids.shape == (int(batch.legal_offsets[-1]),)
    expected_legal_ids = np.concatenate(env.used_legal_ids_history[: batch.T], axis=0)
    assert np.array_equal(batch.legal_ids, expected_legal_ids)

    pass_action_id = resolve_pass_action_id()
    for t in range(batch.T):
        logits = _policy_logits(batch.obs[t], batch.to_play_seat[t])
        legal_ids, legal_offsets = _step_legal_slice(batch, t)
        recomputed = masked_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            batch.action[t].astype(np.int64, copy=False),
            pass_action_id=pass_action_id,
        )
        assert np.array_equal(recomputed, batch.behavior_logp[t])


def test_actor_worker_ids_offsets_snapshots_reused_batch_before_step() -> None:
    worker = ActorWorker(
        actor_id=17,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=31,
    )

    batch = worker.run_once(env=ReusingIdsEnv(), policy_logits_fn=_uniform_policy_logits)

    assert np.array_equal(
        batch.obs[0],
        np.repeat(np.array([[10], [11]], dtype=np.int16), OBS_LEN, axis=1),
    )
    assert np.array_equal(
        batch.obs[1],
        np.repeat(np.array([[17], [18]], dtype=np.int16), OBS_LEN, axis=1),
    )
    assert np.array_equal(batch.to_play_seat[0], np.array([0, 1], dtype=np.int8))
    assert np.array_equal(batch.to_play_seat[1], np.array([1, 0], dtype=np.int8))
    assert np.array_equal(batch.decision_id[0], np.array([100, 100], dtype=np.int32))
    assert np.array_equal(batch.decision_id[1], np.array([101, 101], dtype=np.int32))
    assert np.array_equal(batch.episode_seed[0], np.array([30_000, 30_001], dtype=np.uint64))
    assert np.array_equal(batch.episode_seed[1], np.array([30_010, 30_011], dtype=np.uint64))
    assert np.array_equal(batch.episode_key[0], np.array([40_000, 40_001], dtype=np.uint64))
    assert np.array_equal(batch.episode_key[1], np.array([40_010, 40_011], dtype=np.uint64))
    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert np.array_equal(batch.legal_ids, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32))
    assert np.array_equal(batch.legal_offsets, np.array([0, 2, 4, 6, 8], dtype=np.uint32))


def test_actor_worker_action_counters_prefer_simulator_main_move_flag() -> None:
    action_space = MAIN_MOVE_BASE + 1
    worker = ActorWorker(
        actor_id=12,
        unroll_length=1,
        num_envs=1,
        action_space=action_space,
        layout_name="i16_legal_ids",
        seed=37,
    )

    def logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
        _ = (obs, to_play)
        return np.zeros((1, action_space), dtype=np.float32)

    batch = worker.run_once(env=MainMoveIdWithFalseFlagEnv(), policy_logits_fn=logits)

    assert int(batch.action[0, 0]) == MAIN_MOVE_BASE
    assert batch.counters is not None
    assert batch.counters["main_move_actions"] == 0
    assert batch.counters["max_consecutive_main_moves"] == 0


def test_actor_worker_real_env_ids_offsets_propagates_episode_identity_and_trimmed_legality() -> None:
    worker = ActorWorker(
        actor_id=10,
        unroll_length=2,
        num_envs=2,
        action_space=52,
        layout_name="i16_legal_ids",
        seed=31,
    )
    env = _make_real_env(legality="ids_offsets", num_envs=2, seed=131)
    expected_episode_identity: list[tuple[np.ndarray, np.ndarray]] = []

    def uniform_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
        _ = (obs, to_play)
        expected_episode_identity.append(_pool_episode_identity(env))
        return np.zeros((obs.shape[0], 52), dtype=np.float32)

    try:
        batch = worker.run_once(env=env, policy_logits_fn=uniform_policy_logits)
    finally:
        env.close()

    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None
    assert batch.legal_ids.shape == (int(batch.legal_offsets[-1]),)
    assert batch.episode_seed.dtype == np.uint64
    assert batch.episode_key.dtype == np.uint64
    assert len(expected_episode_identity) == batch.T

    for t, (expected_seed, expected_key) in enumerate(expected_episode_identity):
        assert np.array_equal(batch.episode_seed[t], expected_seed)
        assert np.array_equal(batch.episode_key[t], expected_key)
