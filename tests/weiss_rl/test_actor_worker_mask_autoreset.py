from __future__ import annotations

import numpy as np
from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.core.masking import masked_logp_from_mask, resolve_pass_action_id

from .actor_worker_test_support import (
    ACTION_SPACE,
    OBS_LEN,
    AutoResetMaskEnv,
    FakeMaskEnv,
    PartialDoneResetMaskEnv,
    _expected_obs,
    _policy_logits,
    _uniform_policy_logits,
)


def test_actor_worker_mask_layout_returns_entropy_and_pass_fallback() -> None:
    T = 3
    N = 2
    worker = ActorWorker(
        actor_id=1,
        unroll_length=T,
        num_envs=N,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=5,
    )

    batch = worker.run_once(env=FakeMaskEnv(2), policy_logits_fn=_policy_logits)

    assert batch.legal_mask is not None
    assert batch.legal_mask.shape == (T, N, ACTION_SPACE)
    assert batch.legal_ids is None
    assert batch.legal_offsets is None
    assert batch.entropy is not None
    assert batch.entropy.shape == (T, N)

    env_ids_i32 = np.arange(N, dtype=np.int32)
    env_ids_u64 = np.arange(N, dtype=np.uint64)
    for t in range(T):
        assert np.array_equal(batch.obs[t], _expected_obs(t, num_envs=N))
        assert np.array_equal(batch.to_play_seat[t], ((env_ids_i32 + t) % 2).astype(np.int8))
        assert np.array_equal(batch.decision_id[t], np.full((N,), t, dtype=np.int32))
        assert np.array_equal(batch.reward[t], np.float32(t + 51) + env_ids_i32.astype(np.float32) * np.float32(0.25))
        assert np.array_equal(batch.terminated[t], (t + 1 + env_ids_i32) % 2 == 0)
        assert np.array_equal(batch.truncated[t], (t + 1 + env_ids_i32) % 3 == 1)
        assert np.array_equal(batch.engine_status[t], 700 + (t + 1) * 10 + env_ids_i32)
        assert np.array_equal(batch.episode_seed[t], np.uint64(30_000 + t * 10) + env_ids_u64)
        assert np.array_equal(batch.episode_key[t], np.uint64(40_000 + t * 10) + env_ids_u64)

    pass_action_id = resolve_pass_action_id()
    recomputed = masked_logp_from_mask(
        _policy_logits(batch.obs.reshape(T * N, OBS_LEN), batch.to_play_seat.reshape(T * N)),
        batch.legal_mask.reshape(T * N, ACTION_SPACE),
        batch.action.reshape(T * N).astype(np.int64, copy=False),
        pass_action_id=pass_action_id,
    )

    assert np.allclose(recomputed.reshape(T, N), batch.behavior_logp, atol=0.0, rtol=0.0)
    assert np.all(batch.action[:, 1] == pass_action_id)
    assert np.all(batch.behavior_logp[:, 1] == 0.0)
    assert batch.counters is not None
    assert batch.counters["empty_legal"] == batch.T
    assert batch.counters["engine_fault_done_rows"] == 4
    assert batch.counters["natural_timeout_rows"] == 0
    assert batch.counters["no_progress_timeout_rows"] == 0


def test_actor_worker_keeps_episode_identity_on_terminal_autoreset_transition() -> None:
    worker = ActorWorker(
        actor_id=2,
        unroll_length=2,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=13,
    )
    env = AutoResetMaskEnv()

    batch = worker.run_once(env=env, policy_logits_fn=_uniform_policy_logits)

    assert len(env.reset_done_calls) == 1
    assert np.array_equal(env.reset_done_calls[0], np.array([True], dtype=np.bool_))
    assert np.array_equal(batch.obs[:, 0, 0], np.array([7, 99], dtype=np.int16))
    assert np.array_equal(batch.decision_id[:, 0], np.array([7, 0], dtype=np.int32))
    assert np.array_equal(batch.reward[:, 0], np.array([1.5, 0.25], dtype=np.float32))
    assert np.array_equal(batch.terminated[:, 0], np.array([True, False], dtype=np.bool_))
    assert np.array_equal(batch.truncated[:, 0], np.array([False, False], dtype=np.bool_))
    assert np.array_equal(batch.engine_status[:, 0], np.array([901, 902], dtype=np.int32))
    assert np.array_equal(batch.episode_seed[:, 0], np.array([101, 102], dtype=np.uint64))
    assert np.array_equal(batch.episode_key[:, 0], np.array([201, 202], dtype=np.uint64))


def test_actor_worker_resets_only_done_rows_before_next_policy_action() -> None:
    worker = ActorWorker(
        actor_id=4,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=19,
    )
    env = PartialDoneResetMaskEnv()

    seen_obs: list[np.ndarray] = []

    def policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
        _ = to_play
        seen_obs.append(obs.copy())
        return np.zeros((obs.shape[0], ACTION_SPACE), dtype=np.float32)

    batch = worker.run_once(env=env, policy_logits_fn=policy_logits)

    assert len(seen_obs) == 2
    assert np.array_equal(seen_obs[0][:, 0], np.array([10, 20], dtype=np.int16))
    assert np.array_equal(seen_obs[1][:, 0], np.array([30, 21], dtype=np.int16))
    assert len(env.reset_done_calls) == 1
    assert np.array_equal(env.reset_done_calls[0], np.array([True, False], dtype=np.bool_))
    assert np.array_equal(batch.obs[:, :, 0], np.array([[10, 20], [30, 21]], dtype=np.int16))
    assert np.array_equal(batch.decision_id[1], np.array([5, 1], dtype=np.int32))
    assert np.array_equal(batch.episode_seed[1], np.array([300, 200], dtype=np.uint64))
