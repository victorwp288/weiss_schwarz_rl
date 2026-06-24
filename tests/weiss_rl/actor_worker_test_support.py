from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from weiss_rl.diagnostics.probes.action_diagnostics import MAIN_MOVE_BASE
from weiss_rl.envs.decision_env import DecisionBoundaryEnv

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
        self.main_move_action = np.zeros((num_envs,), dtype=np.bool_)
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


class ReusingIdsEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.step_index = 0
        self.batch = IdsBatch(self.num_envs)
        self.legal_ids = np.zeros((4,), dtype=np.int32)
        self.legal_offsets = np.array([0, 2, 4], dtype=np.uint32)
        self.batch.ids_offsets = (self.legal_ids, self.legal_offsets)

    def reset(self) -> IdsBatch:
        self.step_index = 0
        self._fill_batch()
        return self.batch

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert actions.shape == (self.num_envs,)
        self.step_index += 1
        self._fill_batch()
        return self.batch

    def _fill_batch(self) -> None:
        env_ids_i32 = np.arange(self.num_envs, dtype=np.int32)
        env_ids_u64 = np.arange(self.num_envs, dtype=np.uint64)
        self.batch.obs[:] = np.int16(10 + self.step_index * 7) + env_ids_i32[:, None].astype(np.int16)
        self.batch.to_play[:] = ((env_ids_i32 + self.step_index) % 2).astype(np.int8)
        self.batch.decision_id[:] = np.int32(100 + self.step_index)
        self.batch.reward[:] = np.float32(self.step_index)
        self.batch.terminated[:] = False
        self.batch.truncated[:] = False
        self.batch.engine_status[:] = 0
        self.batch.episode_seed[:] = np.uint64(30_000 + self.step_index * 10) + env_ids_u64
        self.batch.episode_key[:] = np.uint64(40_000 + self.step_index * 10) + env_ids_u64
        self.legal_ids[:] = np.array(
            [
                (self.step_index * 4) % ACTION_SPACE,
                (self.step_index * 4 + 1) % ACTION_SPACE,
                (self.step_index * 4 + 2) % ACTION_SPACE,
                (self.step_index * 4 + 3) % ACTION_SPACE,
            ],
            dtype=np.int32,
        )


class MainMoveIdWithFalseFlagEnv:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self) -> IdsBatch:
        self.step_index = 0
        return self._batch()

    def step(self, actions: np.ndarray) -> IdsBatch:
        assert np.array_equal(actions, np.array([MAIN_MOVE_BASE], dtype=np.uint32))
        self.step_index += 1
        batch = self._batch()
        batch.main_move_action[:] = False
        return batch

    def _batch(self) -> IdsBatch:
        batch = IdsBatch(1)
        batch.obs[:] = self.step_index
        batch.decision_id[:] = self.step_index
        batch.ids_offsets = (
            np.array([MAIN_MOVE_BASE], dtype=np.int32),
            np.array([0, 1], dtype=np.uint32),
        )
        return batch


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
        self.main_move_action = np.zeros((num_envs,), dtype=np.bool_)
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


class NoIdentityIdsBatch:
    def __init__(self, num_envs: int) -> None:
        self.obs = np.zeros((num_envs, OBS_LEN), dtype=np.int16)
        self.reward = np.zeros((num_envs,), dtype=np.float32)
        self.terminated = np.zeros((num_envs,), dtype=np.bool_)
        self.truncated = np.zeros((num_envs,), dtype=np.bool_)
        self.engine_status = np.zeros((num_envs,), dtype=np.int32)
        self.decision_id = np.zeros((num_envs,), dtype=np.int32)
        self.to_play = np.zeros((num_envs,), dtype=np.int8)
        self.actor = self.to_play
        self.ids_offsets: tuple[np.ndarray, np.ndarray] | None = None


class ReplayIdsMissingIdentityAutoResetEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.reset_done_calls: list[np.ndarray] = []

    def reset(self) -> NoIdentityIdsBatch:
        self.step_index = 0
        return self._decision_batch(decision_id=0, obs_value=1)

    def step(self, actions: np.ndarray) -> NoIdentityIdsBatch:
        assert actions.shape == (1,)
        if self.step_index == 0:
            self.step_index = 1
            return self._transition_batch(
                reward=1.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                decision_id=1,
                obs_value=9,
            )
        if self.step_index == 1:
            self.step_index = 2
            return self._transition_batch(
                reward=2.0,
                terminated=True,
                truncated=False,
                engine_status=0,
                decision_id=6,
                obs_value=20,
            )
        raise AssertionError("unexpected extra step")

    def reset_done(self, done: np.ndarray) -> NoIdentityIdsBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        self.reset_done_calls.append(done_array.copy())
        assert np.array_equal(done_array, np.array([True], dtype=np.bool_))
        return self._decision_batch(decision_id=5, obs_value=12)

    def _decision_batch(self, *, decision_id: int, obs_value: int) -> NoIdentityIdsBatch:
        batch = NoIdentityIdsBatch(1)
        batch.obs[:] = obs_value
        batch.decision_id[:] = decision_id
        batch.to_play[:] = 0
        batch.actor[:] = 0
        batch.ids_offsets = (np.array([1, 3, 5], dtype=np.int32), np.array([0, 3], dtype=np.uint32))
        return batch

    def _transition_batch(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        engine_status: int,
        decision_id: int,
        obs_value: int,
    ) -> NoIdentityIdsBatch:
        batch = self._decision_batch(decision_id=decision_id, obs_value=obs_value)
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
