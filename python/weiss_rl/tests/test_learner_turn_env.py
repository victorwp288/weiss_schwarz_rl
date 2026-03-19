from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from weiss_rl.envs.learner_turn_env import LearnerTurnEnv


def _batch(
    to_play_seat: list[int],
    *,
    reward: list[float] | None = None,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
) -> SimpleNamespace:
    batch_size = len(to_play_seat)
    reward_arr = np.zeros((batch_size,), dtype=np.float32) if reward is None else np.asarray(reward, dtype=np.float32)
    terminated_arr = np.zeros((batch_size,), dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    truncated_arr = np.zeros((batch_size,), dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool)
    return SimpleNamespace(
        to_play_seat=np.asarray(to_play_seat, dtype=np.int8),
        reward=reward_arr,
        terminated=terminated_arr,
        truncated=truncated_arr,
    )


class FakeEnv:
    def __init__(self, reset_batch: SimpleNamespace, scripted_steps: list[tuple[np.ndarray, SimpleNamespace]]) -> None:
        self._reset_batch = reset_batch
        self._scripted_steps = list(scripted_steps)
        self.reset_seed: int | None = None
        self.seen_actions: list[np.ndarray] = []

    def reset(self, seed: int | None = None) -> SimpleNamespace:
        self.reset_seed = seed
        return self._reset_batch

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        if not self._scripted_steps:
            raise AssertionError("unexpected extra step")

        actual_actions = np.asarray(actions).copy()
        expected_actions, next_batch = self._scripted_steps.pop(0)
        self.seen_actions.append(actual_actions)
        npt.assert_array_equal(actual_actions, expected_actions)
        return next_batch


def test_step_uses_stored_batch_and_folds_opponent_turns_until_learner_turn_returns() -> None:
    reset_batch = _batch([0, 0])
    after_learner_step = _batch([1, 1], reward=[1.5, -2.0])
    after_opponent_step = _batch([0, 0], reward=[0.25, 0.5])
    env = FakeEnv(
        reset_batch,
        scripted_steps=[
            (np.array([10, 11], dtype=np.int64), after_learner_step),
            (np.array([20, 21], dtype=np.int64), after_opponent_step),
        ],
    )

    def opponent_policy(batch: SimpleNamespace, opponent_mask: np.ndarray) -> np.ndarray:
        npt.assert_array_equal(batch.to_play_seat, np.array([1, 1], dtype=np.int8))
        npt.assert_array_equal(opponent_mask, np.array([True, True]))
        return np.array([20, 21], dtype=np.int64)

    wrapped = LearnerTurnEnv(env, learner_seat=0, opponent_policy=opponent_policy)

    reset = wrapped.reset(seed=123)
    batch, reward_learn, done, info = wrapped.step(np.array([10, 11], dtype=np.int64))

    assert reset is reset_batch
    assert batch is after_opponent_step
    assert env.reset_seed == 123
    npt.assert_allclose(reward_learn, np.array([1.25, -2.5], dtype=np.float32))
    npt.assert_array_equal(done, np.array([False, False]))
    npt.assert_array_equal(info.k_raw_decisions, np.array([2, 2], dtype=np.int32))
    npt.assert_array_equal(info.terminal_during_opponent_internal, np.array([False, False]))


def test_step_can_start_on_opponent_turn_and_marks_terminal_during_internal_opponent() -> None:
    reset_batch = _batch([1, 1])
    after_opponent_step = _batch([0, 0], reward=[0.75, 1.25], terminated=[True, False])
    env = FakeEnv(
        reset_batch,
        scripted_steps=[
            (np.array([30, 31], dtype=np.int64), after_opponent_step),
        ],
    )

    def opponent_policy(batch: SimpleNamespace, opponent_mask: np.ndarray) -> np.ndarray:
        npt.assert_array_equal(batch.to_play_seat, np.array([1, 1], dtype=np.int8))
        npt.assert_array_equal(opponent_mask, np.array([True, True]))
        return np.array([30, 31], dtype=np.int64)

    wrapped = LearnerTurnEnv(env, learner_seat=0, opponent_policy=opponent_policy)
    wrapped.reset()

    batch, reward_learn, done, info = wrapped.step(np.array([99, 98], dtype=np.int64))

    assert batch is after_opponent_step
    npt.assert_allclose(reward_learn, np.array([-0.75, -1.25], dtype=np.float32))
    npt.assert_array_equal(done, np.array([True, False]))
    npt.assert_array_equal(info.k_raw_decisions, np.array([1, 1], dtype=np.int32))
    npt.assert_array_equal(info.terminal_during_opponent_internal, np.array([True, False]))


def test_step_requires_reset_first() -> None:
    wrapped = LearnerTurnEnv(FakeEnv(_batch([0]), []), learner_seat=0, opponent_policy=lambda *_: np.array([0]))

    with pytest.raises(RuntimeError, match=r"reset\(\) must be called before step\(\)"):
        wrapped.step(np.array([7], dtype=np.int64))


def test_step_rejects_invalid_opponent_policy_shape() -> None:
    wrapped = LearnerTurnEnv(
        FakeEnv(_batch([1, 1]), []),
        learner_seat=0,
        opponent_policy=lambda *_: np.array([7], dtype=np.int64),
    )
    batch = wrapped.reset()

    with pytest.raises(ValueError, match=r"opponent_policy must return actions shaped \[B\]"):
        wrapped.step_from_batch(batch, np.array([1, 2], dtype=np.int64))


def test_step_raises_when_batched_rows_diverge_after_some_return_to_learner_turn() -> None:
    reset_batch = _batch([0, 1])
    after_first_raw_step = _batch([1, 0], reward=[1.0, 0.5])
    env = FakeEnv(
        reset_batch,
        scripted_steps=[
            (np.array([10, 30], dtype=np.int64), after_first_raw_step),
        ],
    )

    def opponent_policy(_: SimpleNamespace, opponent_mask: np.ndarray) -> np.ndarray:
        npt.assert_array_equal(opponent_mask, np.array([False, True]))
        return np.array([0, 30], dtype=np.int64)

    wrapped = LearnerTurnEnv(env, learner_seat=0, opponent_policy=opponent_policy)
    wrapped.reset()

    with pytest.raises(RuntimeError, match="cannot safely fold a diverged batch"):
        wrapped.step(np.array([10, 11], dtype=np.int64))
