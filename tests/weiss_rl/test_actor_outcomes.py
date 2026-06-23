from __future__ import annotations

import numpy as np
from weiss_rl.actors.actor_worker import ActorWorker

OBS_LEN = 4
ACTION_SPACE = 8


class OutcomeBatch:
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
        self.masks = np.ones((num_envs, ACTION_SPACE), dtype=np.uint8)
        self.opponent_policy_id = np.empty((num_envs,), dtype=object)


class OpponentTrackingEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.reset_done_calls: list[np.ndarray] = []

    def reset(self) -> OutcomeBatch:
        self.step_index = 0
        return self._batch(
            obs=np.array([10, 20], dtype=np.int16),
            rewards=np.array([0.0, 0.0], dtype=np.float32),
            terminated=np.array([False, False], dtype=np.bool_),
            truncated=np.array([False, False], dtype=np.bool_),
            decision_id=np.array([0, 0], dtype=np.int32),
            episode_seed=np.array([100, 200], dtype=np.uint64),
            episode_key=np.array([1000, 2000], dtype=np.uint64),
            opponent_policy_id=np.array(["opp_a", "opp_b"], dtype=object),
        )

    def step(self, actions: np.ndarray) -> OutcomeBatch:
        assert actions.shape == (2,)
        self.step_index += 1
        if self.step_index == 1:
            return self._batch(
                obs=np.array([99, 21], dtype=np.int16),
                rewards=np.array([1.0, 0.0], dtype=np.float32),
                terminated=np.array([True, False], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([1, 1], dtype=np.int32),
                episode_seed=np.array([101, 200], dtype=np.uint64),
                episode_key=np.array([1001, 2000], dtype=np.uint64),
                opponent_policy_id=np.array(["opp_a", "opp_b"], dtype=object),
            )
        if self.step_index == 2:
            return self._batch(
                obs=np.array([31, 22], dtype=np.int16),
                rewards=np.array([0.0, -1.0], dtype=np.float32),
                terminated=np.array([True, True], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([6, 2], dtype=np.int32),
                episode_seed=np.array([300, 200], dtype=np.uint64),
                episode_key=np.array([3000, 2000], dtype=np.uint64),
                opponent_policy_id=np.array(["opp_c", "opp_b"], dtype=object),
            )
        raise AssertionError("unexpected extra step")

    def reset_done(self, done: np.ndarray) -> OutcomeBatch:
        done_array = np.asarray(done, dtype=np.bool_)
        self.reset_done_calls.append(done_array.copy())
        if np.array_equal(done_array, np.array([True, False], dtype=np.bool_)):
            return self._batch(
                obs=np.array([30, 21], dtype=np.int16),
                rewards=np.array([0.0, 0.0], dtype=np.float32),
                terminated=np.array([False, False], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([5, 1], dtype=np.int32),
                episode_seed=np.array([300, 200], dtype=np.uint64),
                episode_key=np.array([3000, 2000], dtype=np.uint64),
                opponent_policy_id=np.array(["opp_c", None], dtype=object),
            )
        if np.array_equal(done_array, np.array([True, True], dtype=np.bool_)):
            return self._batch(
                obs=np.array([40, 50], dtype=np.int16),
                rewards=np.array([0.0, 0.0], dtype=np.float32),
                terminated=np.array([False, False], dtype=np.bool_),
                truncated=np.array([False, False], dtype=np.bool_),
                decision_id=np.array([0, 0], dtype=np.int32),
                episode_seed=np.array([400, 500], dtype=np.uint64),
                episode_key=np.array([4000, 5000], dtype=np.uint64),
                opponent_policy_id=np.array(["opp_d", "opp_e"], dtype=object),
            )
        raise AssertionError(f"unexpected reset_done mask: {done_array}")

    def _batch(
        self,
        *,
        obs: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        decision_id: np.ndarray,
        episode_seed: np.ndarray,
        episode_key: np.ndarray,
        opponent_policy_id: np.ndarray,
    ) -> OutcomeBatch:
        batch = OutcomeBatch(2)
        batch.obs[:] = np.repeat(obs[:, None], OBS_LEN, axis=1)
        batch.rewards[:] = rewards
        batch.terminated[:] = terminated
        batch.truncated[:] = truncated
        batch.decision_id[:] = decision_id
        batch.episode_seed[:] = episode_seed
        batch.episode_key[:] = episode_key
        batch.opponent_policy_id[:] = opponent_policy_id
        return batch


def _uniform_policy_logits(obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    _ = (obs, to_play)
    return np.zeros((obs.shape[0], ACTION_SPACE), dtype=np.float32)


def test_actor_worker_tracks_outcomes_by_initial_and_reset_done_opponent_ids() -> None:
    worker = ActorWorker(
        actor_id=0,
        unroll_length=2,
        num_envs=2,
        action_space=ACTION_SPACE,
        layout_name="mask",
        seed=7,
    )
    env = OpponentTrackingEnv()

    worker.run_once(env=env, policy_logits_fn=_uniform_policy_logits)

    assert len(env.reset_done_calls) == 2
    assert np.array_equal(env.reset_done_calls[0], np.array([True, False], dtype=np.bool_))
    assert np.array_equal(env.reset_done_calls[1], np.array([True, True], dtype=np.bool_))
    assert worker.outcomes.counts("opp_a") == (1, 0, 0, 0)
    assert worker.outcomes.counts("opp_b") == (0, 1, 0, 0)
    assert worker.outcomes.counts("opp_c") == (0, 0, 1, 0)
    assert all(opponent_id != "unknown" for _epoch, opponent_id in worker.outcomes.by_opponent)
