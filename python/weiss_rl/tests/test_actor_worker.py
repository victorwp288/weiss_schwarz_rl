from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from weiss_rl.actors.actor_worker import ActorWorker
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.league.opponent_pool import OpponentPoolSampler, sample_opponent_snapshot_ids
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.masking import (
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
    resolve_pass_action_id,
    sample_actions_from_mask,
)
from weiss_rl.replay.bundles import load_replay_bundle

OBS_LEN = 6
ACTION_SPACE = 64
_REAL_SIM_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


def _load_weiss_sim():
    return pytest.importorskip("weiss_sim")


def _simulator_episode_key(
    episode_seed: np.ndarray,
    episode_index: np.ndarray,
    env_index: np.ndarray,
) -> np.ndarray:
    runner = pytest.importorskip("weiss_sim.runner")
    return runner._episode_key(
        np.asarray(episode_seed, dtype=np.uint64),
        np.asarray(episode_index, dtype=np.uint64),
        np.asarray(env_index, dtype=np.uint64),
    )


def _pool_episode_identity(env: DecisionBoundaryEnv) -> tuple[np.ndarray, np.ndarray]:
    episode_seed = env.pool.episode_seed_batch().astype(np.uint64, copy=False)
    episode_index = env.pool.episode_index_batch().astype(np.uint64, copy=False)
    env_index = env.pool.env_index_batch().astype(np.uint64, copy=False)
    episode_key = _simulator_episode_key(episode_seed, episode_index, env_index)
    return episode_seed.copy(), episode_key.copy()


def _make_real_env(
    *,
    legality: Literal["mask", "ids_offsets"],
    num_envs: int = 2,
    seed: int = 123,
) -> DecisionBoundaryEnv:
    _load_weiss_sim()
    return DecisionBoundaryEnv.create(
        legality=legality,
        mode="train",
        num_envs=num_envs,
        db_path=None,
        deck_lists=[_REAL_SIM_LEGAL_DECK, _REAL_SIM_LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=seed,
    )


class IdsBatch:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, OBS_LEN), dtype=np.int16)
        self.reward = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=np.bool_)
        self.truncated = np.zeros((num_envs,), dtype=np.bool_)
        self.engine_status = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int32)
        self.to_play = np.zeros((num_envs,), dtype=np.int8)
        self.actor = self.to_play
        self.episode_seed = np.zeros((num_envs,), dtype=np.uint64)
        self.episode_key = np.zeros((num_envs,), dtype=np.uint64)
        self.ids_offsets: tuple[np.ndarray, np.ndarray] | None = None


class FakeIdsEnv:
    def __init__(self, num_envs: int, *, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.step_index = 0

    def reset(self) -> IdsBatch:
        self.step_index = 0
        return self._batch()

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert actions.shape == (self.num_envs,)
        self.step_index += 1
        return self._batch()

    def _batch(self) -> IdsBatch:
        batch = IdsBatch(self.num_envs)
        env_ids_i32 = np.arange(self.num_envs, dtype=np.int32)
        env_ids_u64 = np.arange(self.num_envs, dtype=np.uint64)

        batch.obs[:] = (self.step_index + env_ids_i32[:, None]).astype(np.int16)
        batch.reward[:] = np.float32(self.step_index) + env_ids_i32.astype(np.float32) * np.float32(0.1)
        batch.terminated[:] = (self.step_index + env_ids_i32) % 2 == 1
        batch.truncated[:] = (self.step_index + env_ids_i32) % 3 == 2
        batch.engine_status[:] = 100 + self.step_index * 10 + env_ids_i32
        batch.to_play[:] = ((env_ids_i32 + self.step_index) % 2).astype(np.int8)
        batch.decision_id[:] = self.step_index
        batch.episode_seed[:] = np.uint64(10_000 + self.step_index * 10) + env_ids_u64
        batch.episode_key[:] = np.uint64(20_000 + self.step_index * 10) + env_ids_u64

        legal_slices = []
        offsets = [0]
        for env_index in range(self.num_envs):
            if env_index == 0 and self.step_index % 2 == 0:
                legal_ids = np.array([], dtype=np.int32)
            else:
                size = 2 + env_index
                legal_ids = np.sort(self.rng.choice(ACTION_SPACE, size=size, replace=False)).astype(np.int32)
            legal_slices.append(legal_ids)
            offsets.append(offsets[-1] + int(legal_ids.size))

        batch.ids_offsets = self._ids_offsets_payload(legal_slices, offsets)
        return batch

    def _ids_offsets_payload(
        self,
        legal_slices: list[np.ndarray],
        offsets: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate(legal_slices, axis=0) if offsets[-1] else np.array([], dtype=np.int32),
            np.array(offsets, dtype=np.uint32),
        )


class PaddedIdsEnv(FakeIdsEnv):
    def __init__(self, num_envs: int, *, seed: int = 0) -> None:
        super().__init__(num_envs, seed=seed)
        self.used_legal_ids_history: list[np.ndarray] = []
        self.used_legal_offsets_history: list[np.ndarray] = []

    def _ids_offsets_payload(
        self,
        legal_slices: list[np.ndarray],
        offsets: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        legal_ids, legal_offsets = super()._ids_offsets_payload(legal_slices, offsets)
        self.used_legal_ids_history.append(legal_ids.copy())
        self.used_legal_offsets_history.append(legal_offsets.copy())
        padding = np.full((8,), ACTION_SPACE - 1, dtype=np.int32)
        return np.concatenate((legal_ids, padding), axis=0), legal_offsets


class MaskBatch:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, OBS_LEN), dtype=np.int16)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=np.bool_)
        self.truncated = np.zeros((num_envs,), dtype=np.bool_)
        self.engine_status = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int32)
        self.actor = np.zeros((num_envs,), dtype=np.int8)
        self.episode_seed = np.zeros((num_envs,), dtype=np.uint64)
        self.episode_key = np.zeros((num_envs,), dtype=np.uint64)
        self.masks = np.zeros((num_envs, ACTION_SPACE), dtype=np.uint8)


class FakeMaskEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.step_index = 0

    def reset(self) -> MaskBatch:
        self.step_index = 0
        return self._batch()

    def step(self, actions: np.ndarray) -> MaskBatch:
        assert actions.shape == (self.num_envs,)
        self.step_index += 1
        return self._batch()

    def _batch(self) -> MaskBatch:
        batch = MaskBatch(self.num_envs)
        env_ids_i32 = np.arange(self.num_envs, dtype=np.int32)
        env_ids_u64 = np.arange(self.num_envs, dtype=np.uint64)

        batch.obs[:] = (self.step_index + env_ids_i32[:, None]).astype(np.int16)
        batch.rewards[:] = np.float32(self.step_index + 50) + env_ids_i32.astype(np.float32) * np.float32(0.25)
        batch.terminated[:] = (self.step_index + env_ids_i32) % 2 == 0
        batch.truncated[:] = (self.step_index + env_ids_i32) % 3 == 1
        batch.engine_status[:] = 700 + self.step_index * 10 + env_ids_i32
        batch.actor[:] = ((env_ids_i32 + self.step_index) % 2).astype(np.int8)
        batch.decision_id[:] = self.step_index
        batch.episode_seed[:] = np.uint64(30_000 + self.step_index * 10) + env_ids_u64
        batch.episode_key[:] = np.uint64(40_000 + self.step_index * 10) + env_ids_u64
        batch.masks[0, [1, 3, 5]] = 1
        batch.masks[1, :] = 0
        return batch


class StaticMaskEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def reset(self) -> MaskBatch:
        return self._batch()

    def step(self, actions: np.ndarray) -> MaskBatch:
        assert actions.shape == (self.num_envs,)
        return self._batch()

    def _batch(self) -> MaskBatch:
        batch = MaskBatch(self.num_envs)
        batch.masks[:] = 1
        return batch


class AutoResetMaskEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.reset_done_calls: list[np.ndarray] = []

    def reset(self) -> MaskBatch:
        self.step_index = 0
        return self._decision_batch(episode_seed=101, episode_key=201, decision_id=7, obs_value=7)

    def step(self, actions: np.ndarray) -> MaskBatch:
        assert actions.shape == (1,)
        if self.step_index == 0:
            self.step_index = 1
            return self._transition_batch(
                reward=1.5,
                terminated=True,
                truncated=False,
                engine_status=901,
                episode_seed=102,
                episode_key=202,
                decision_id=0,
                obs_value=99,
            )
        self.step_index += 1
        return self._transition_batch(
            reward=0.25,
            terminated=False,
            truncated=False,
            engine_status=902,
            episode_seed=102,
            episode_key=202,
            decision_id=1,
            obs_value=100,
        )

    def reset_done(self, done: np.ndarray) -> MaskBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        self.reset_done_calls.append(done_array.copy())
        assert np.array_equal(done_array, np.array([True], dtype=np.bool_))
        return self._decision_batch(episode_seed=102, episode_key=202, decision_id=0, obs_value=99)

    def _decision_batch(self, *, episode_seed: int, episode_key: int, decision_id: int, obs_value: int) -> MaskBatch:
        batch = MaskBatch(1)
        batch.obs[:] = obs_value
        batch.decision_id[:] = decision_id
        batch.episode_seed[:] = np.uint64(episode_seed)
        batch.episode_key[:] = np.uint64(episode_key)
        batch.masks[:] = 1
        return batch

    def _transition_batch(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        engine_status: int,
        episode_seed: int,
        episode_key: int,
        decision_id: int,
        obs_value: int,
    ) -> MaskBatch:
        batch = self._decision_batch(
            episode_seed=episode_seed,
            episode_key=episode_key,
            decision_id=decision_id,
            obs_value=obs_value,
        )
        batch.rewards[:] = np.float32(reward)
        batch.terminated[:] = terminated
        batch.truncated[:] = truncated
        batch.engine_status[:] = engine_status
        return batch


class PartialDoneResetMaskEnv:
    def __init__(self) -> None:
        self.reset_done_calls: list[np.ndarray] = []
        self.step_index = 0

    def reset(self) -> MaskBatch:
        self.step_index = 0
        return self._batch(
            obs=np.array([10, 20], dtype=np.int16),
            reward=np.array([0.0, 0.0], dtype=np.float32),
            terminated=np.array([False, False], dtype=np.bool_),
            truncated=np.array([False, False], dtype=np.bool_),
            decision_id=np.array([0, 0], dtype=np.int32),
            episode_seed=np.array([100, 200], dtype=np.uint64),
            episode_key=np.array([1000, 2000], dtype=np.uint64),
        )

    def step(self, actions: np.ndarray) -> MaskBatch:
        assert actions.shape == (2,)
        self.step_index += 1
        if self.step_index == 1:
            return self._batch(
                obs=np.array([99, 21], dtype=np.int16),
                reward=np.array([1.0, 2.0], dtype=np.float32),
                terminated=np.array([True, False], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([1, 1], dtype=np.int32),
                episode_seed=np.array([101, 200], dtype=np.uint64),
                episode_key=np.array([1001, 2000], dtype=np.uint64),
            )
        if self.step_index == 2:
            return self._batch(
                obs=np.array([31, 22], dtype=np.int16),
                reward=np.array([3.0, 4.0], dtype=np.float32),
                terminated=np.array([False, False], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([6, 2], dtype=np.int32),
                episode_seed=np.array([300, 200], dtype=np.uint64),
                episode_key=np.array([3000, 2000], dtype=np.uint64),
            )
        raise AssertionError("unexpected extra step")

    def reset_done(self, done: np.ndarray) -> MaskBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        self.reset_done_calls.append(done_array.copy())
        assert np.array_equal(done_array, np.array([True, False], dtype=np.bool_))
        return self._batch(
            obs=np.array([30, 21], dtype=np.int16),
            reward=np.array([0.0, 2.0], dtype=np.float32),
            terminated=np.array([False, False], dtype=np.bool_),
            truncated=np.array([False, False], dtype=np.bool_),
            decision_id=np.array([5, 1], dtype=np.int32),
            episode_seed=np.array([300, 200], dtype=np.uint64),
            episode_key=np.array([3000, 2000], dtype=np.uint64),
        )

    def _batch(
        self,
        *,
        obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        decision_id: np.ndarray,
        episode_seed: np.ndarray,
        episode_key: np.ndarray,
    ) -> MaskBatch:
        batch = MaskBatch(2)
        batch.obs[:] = np.repeat(obs[:, None], OBS_LEN, axis=1)
        batch.rewards[:] = reward
        batch.terminated[:] = terminated
        batch.truncated[:] = truncated
        batch.decision_id[:] = decision_id
        batch.episode_seed[:] = episode_seed
        batch.episode_key[:] = episode_key
        batch.masks[:] = 1
        return batch


class ReplayIdsAutoResetEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.reset_done_calls: list[np.ndarray] = []

    def reset(self) -> IdsBatch:
        self.step_index = 0
        return self._decision_batch(episode_seed=11, episode_key=111, decision_id=0, obs_value=1)

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert actions.shape == (1,)
        if self.step_index == 0:
            self.step_index = 1
            return self._transition_batch(
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                episode_seed=12,
                episode_key=112,
                decision_id=1,
                obs_value=9,
            )
        if self.step_index == 1:
            self.step_index = 2
            return self._transition_batch(
                reward=2.0,
                terminated=False,
                truncated=False,
                engine_status=0,
                episode_seed=22,
                episode_key=222,
                decision_id=6,
                obs_value=20,
            )
        raise AssertionError("unexpected extra step")

    def reset_done(self, done: np.ndarray) -> IdsBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        self.reset_done_calls.append(done_array.copy())
        assert np.array_equal(done_array, np.array([True], dtype=np.bool_))
        return self._decision_batch(episode_seed=22, episode_key=222, decision_id=5, obs_value=12)

    def _decision_batch(self, *, episode_seed: int, episode_key: int, decision_id: int, obs_value: int) -> IdsBatch:
        batch = IdsBatch(1)
        batch.obs[:] = obs_value
        batch.decision_id[:] = decision_id
        batch.to_play[:] = 0
        batch.actor[:] = 0
        batch.episode_seed[:] = np.uint64(episode_seed)
        batch.episode_key[:] = np.uint64(episode_key)
        batch.ids_offsets = (np.array([1, 3, 5], dtype=np.int32), np.array([0, 3], dtype=np.uint32))
        return batch

    def _transition_batch(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        engine_status: int,
        episode_seed: int,
        episode_key: int,
        decision_id: int,
        obs_value: int,
    ) -> IdsBatch:
        batch = self._decision_batch(
            episode_seed=episode_seed,
            episode_key=episode_key,
            decision_id=decision_id,
            obs_value=obs_value,
        )
        batch.reward[:] = np.float32(reward)
        batch.terminated[:] = terminated
        batch.truncated[:] = truncated
        batch.engine_status[:] = engine_status
        return batch


class EngineFaultIdsEnv:
    def reset(self) -> IdsBatch:
        return self._decision_batch(episode_seed=77, episode_key=777, decision_id=0, obs_value=4)

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert actions.shape == (1,)
        return self._transition_batch(
            reward=-1.0,
            terminated=True,
            truncated=False,
            engine_status=17,
            episode_seed=78,
            episode_key=778,
            decision_id=1,
            obs_value=8,
        )

    def _decision_batch(self, *, episode_seed: int, episode_key: int, decision_id: int, obs_value: int) -> IdsBatch:
        batch = IdsBatch(1)
        batch.obs[:] = obs_value
        batch.decision_id[:] = decision_id
        batch.to_play[:] = 0
        batch.actor[:] = 0
        batch.episode_seed[:] = np.uint64(episode_seed)
        batch.episode_key[:] = np.uint64(episode_key)
        batch.ids_offsets = (np.array([2, 4, 6], dtype=np.int32), np.array([0, 3], dtype=np.uint32))
        return batch

    def _transition_batch(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        engine_status: int,
        episode_seed: int,
        episode_key: int,
        decision_id: int,
        obs_value: int,
    ) -> IdsBatch:
        batch = self._decision_batch(
            episode_seed=episode_seed,
            episode_key=episode_key,
            decision_id=decision_id,
            obs_value=obs_value,
        )
        batch.reward[:] = np.float32(reward)
        batch.terminated[:] = terminated
        batch.truncated[:] = truncated
        batch.engine_status[:] = engine_status
        return batch


def _policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    logits = np.zeros((obs.shape[0], ACTION_SPACE), dtype=np.float32)
    base = obs[:, 0].astype(np.float32) + to_play.astype(np.float32)
    logits[:] = base[:, None] * 0.01 + np.arange(ACTION_SPACE, dtype=np.float32)[None, :] * 0.001
    return logits


def _uniform_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    _ = (obs, to_play)
    return np.zeros((obs.shape[0], ACTION_SPACE), dtype=np.float32)


def _step_legal_slice(batch, t: int) -> tuple[np.ndarray, np.ndarray]:
    assert batch.legal_ids is not None
    assert batch.legal_offsets is not None

    row_start = t * batch.N
    row_stop = row_start + batch.N
    offset_start = int(batch.legal_offsets[row_start])
    offset_stop = int(batch.legal_offsets[row_stop])
    legal_ids = batch.legal_ids[offset_start:offset_stop]
    legal_offsets = batch.legal_offsets[row_start : row_stop + 1] - offset_start
    return legal_ids, legal_offsets


def _expected_obs(t: int, *, num_envs: int) -> np.ndarray:
    return np.repeat(
        (t + np.arange(num_envs, dtype=np.int32)[:, None]).astype(np.int16),
        OBS_LEN,
        axis=1,
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
    assert batch.counters == {"empty_legal": batch.T}


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


def test_actor_worker_samples_opponent_policy_ids_on_episode_boundaries() -> None:
    registry = SnapshotRegistry()
    for update, snapshot_id in enumerate(["s1", "s2", "s3"], start=1):
        registry.add_snapshot(
            policy_id=snapshot_id,
            update=update,
            weights_sha256=(snapshot_id * 64)[:64].ljust(64, "0"),
            path=snapshot_weights_relpath(snapshot_id),
        )
    registry.add_champion("s1")
    sampler = OpponentPoolSampler(
        registry=registry,
        recent_size=2,
        champion_size=1,
        win_rates_by_snapshot_id={"s2": 0.9, "s3": 0.2},
    )
    worker = ActorWorker(
        actor_id=5,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=37,
        opponent_sampler=sampler,
    )
    env = PartialDoneResetMaskEnv()
    assignments: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def record_assignment(done: np.ndarray, opponent_policy_ids: tuple[str, ...]) -> None:
        assignments.append((done.copy(), opponent_policy_ids))

    worker.opponent_assignment_fn = record_assignment
    worker.run_once(env=env, policy_logits_fn=_uniform_policy_logits)

    expected_rng = np.random.default_rng(np.random.SeedSequence([worker.seed, worker.actor_id, 1]))
    pool_ids = sampler.snapshot_ids()
    expected_initial = sample_opponent_snapshot_ids(
        pool_ids,
        count=2,
        rng=expected_rng,
        win_rates_by_snapshot_id=sampler.win_rates_by_snapshot_id,
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
        neutral_win_rate=sampler.neutral_win_rate,
    )
    expected_reset = sample_opponent_snapshot_ids(
        pool_ids,
        count=1,
        rng=expected_rng,
        win_rates_by_snapshot_id=sampler.win_rates_by_snapshot_id,
        power=sampler.power,
        eps_uniform=sampler.eps_uniform,
        neutral_win_rate=sampler.neutral_win_rate,
    )

    assert len(assignments) == 2
    assert np.array_equal(assignments[0][0], np.array([True, True], dtype=np.bool_))
    assert assignments[0][1] == expected_initial
    assert np.array_equal(assignments[1][0], np.array([True, False], dtype=np.bool_))
    assert assignments[1][1] == (expected_reset[0], expected_initial[1])
    assert worker.current_opponent_policy_ids == (expected_reset[0], expected_initial[1])


def test_actor_worker_preserves_rng_stream_across_run_once_calls() -> None:
    actor_id = 3
    seed = 17
    unroll_length = 8
    num_envs = 4
    worker = ActorWorker(
        actor_id=actor_id,
        unroll_length=unroll_length,
        num_envs=num_envs,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=seed,
    )

    first = worker.run_once(env=StaticMaskEnv(num_envs), policy_logits_fn=_uniform_policy_logits)
    second = worker.run_once(env=StaticMaskEnv(num_envs), policy_logits_fn=_uniform_policy_logits)

    rng = np.random.default_rng(seed + actor_id)
    logits = np.zeros((num_envs, ACTION_SPACE), dtype=np.float32)
    legal_mask = np.ones((num_envs, ACTION_SPACE), dtype=np.uint8)
    expected_chunks = []
    for _ in range(2):
        chunk_rows = []
        for _ in range(unroll_length):
            actions, _, _ = sample_actions_from_mask(logits, legal_mask, rng=rng)
            chunk_rows.append(actions.astype(np.uint32, copy=False))
        expected_chunks.append(np.stack(chunk_rows, axis=0))

    assert np.array_equal(first.action, expected_chunks[0])
    assert np.array_equal(second.action, expected_chunks[1])


def test_actor_worker_clears_replay_buffer_on_episode_boundary(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    worker = ActorWorker(
        actor_id=12,
        unroll_length=2,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=101,
        replay_dir=replay_dir,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("ab" * 32),
    )

    worker.run_once(env=ReplayIdsAutoResetEnv(), policy_logits_fn=_uniform_policy_logits)
    worker._flush_replay_for_env(env_index=0)

    [bundle_path] = sorted(replay_dir.glob("replay_*.zip"))
    meta, steps, fault = load_replay_bundle(bundle_path)

    assert fault is None
    assert meta.episode_identity_source == "simulator"
    assert meta.simulator_episode_key_u64 == 222
    assert meta.episode_seed64 == 22
    assert [step.decision_id for step in steps] == [5]
    assert [step.t for step in steps] == [1]


def test_actor_worker_captures_engine_error_replay_with_actual_episode_identity(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    worker = ActorWorker(
        actor_id=13,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        layout_name="i16_legal_ids",
        seed=202,
        replay_dir=replay_dir,
        run_id256=b"r" * 32,
        spec_hash256=bytes.fromhex("cd" * 32),
        capture_replays_on_done=False,
    )

    worker.run_once(env=EngineFaultIdsEnv(), policy_logits_fn=_uniform_policy_logits)

    [bundle_path] = sorted(replay_dir.glob("replay_*.zip"))
    meta, steps, fault = load_replay_bundle(bundle_path)

    assert meta.episode_identity_source == "simulator"
    assert meta.simulator_episode_key_u64 == 777
    assert meta.episode_seed64 == 77
    assert len(steps) == 1
    assert steps[0].engine_status == 17
    assert fault is not None
    assert fault["engine_status"] == 17
    assert fault["simulator_episode_key"] == 777


def test_actor_worker_writes_fault_bundle_on_nonfinite_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    worker = ActorWorker(
        actor_id=8,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=41,
        fault_dir=fault_dir,
    )

    def nan_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
        logits = _policy_logits(obs, to_play)
        logits[0, 0] = np.nan
        return logits

    with pytest.raises(RuntimeError, match="non-finite actor policy logits; wrote fault bundle to ") as excinfo:
        worker.run_once(env=StaticMaskEnv(2), policy_logits_fn=nan_policy_logits)

    [fault_path] = sorted(fault_dir.glob("actor_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "actor_worker"
    assert payload["reason"] == "non-finite actor policy logits"
    assert payload["step"] == 0
    assert payload["logits_nonfinite_indices"]["data"] == [[0, 0]]


def test_actor_worker_reports_checkpoint_metadata_lag_in_update_units(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_metadata_100.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_metadata_250.json").write_text("{}\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=7,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=2,
    )

    assert worker.checkpoint_metadata_poll_interval_updates == 2

    first = worker.poll_checkpoint_metadata()
    assert first == {"observed_checkpoint_update": 0, "checkpoint_metadata_lag_updates": 250}

    second = worker.poll_checkpoint_metadata()
    assert second == {"observed_checkpoint_update": 250, "checkpoint_metadata_lag_updates": 0}

    (checkpoint_dir / "checkpoint_metadata_400.json").write_text("{}\n", encoding="utf-8")
    third = worker.poll_checkpoint_metadata()
    assert third == {"observed_checkpoint_update": 250, "checkpoint_metadata_lag_updates": 150}

    fourth = worker.poll_checkpoint_metadata()
    assert fourth == {"observed_checkpoint_update": 400, "checkpoint_metadata_lag_updates": 0}


def test_actor_worker_supports_legacy_checkpoint_filenames_for_metadata_tracking(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_latest.pt").write_text("bad\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_50.pt").write_text("stub\n", encoding="utf-8")

    worker = ActorWorker(
        actor_id=1,
        unroll_length=1,
        num_envs=1,
        action_space=ACTION_SPACE,
        checkpoint_dir=checkpoint_dir,
        reload_interval_updates=1,
    )

    result = worker.poll_checkpoint_metadata()
    assert result == {"observed_checkpoint_update": 50, "checkpoint_metadata_lag_updates": 0}
    assert worker.loaded_checkpoint_update == 50
    assert worker.checkpoint_lag_updates == 0
