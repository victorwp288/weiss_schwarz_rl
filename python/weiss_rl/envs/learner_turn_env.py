"""Optional learner-turn wrapper over decision-boundary steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DecisionBoundaryEnv(Protocol):
    """Minimal protocol for a decision-boundary batched env."""

    def reset(self, seed: int | None = None) -> Any: ...
    def step(self, actions: np.ndarray) -> Any: ...


OpponentPolicy = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(slots=True)
class LearnerTurnStepInfo:
    """Per-row metadata for a folded learner-turn step."""

    k_raw_decisions: np.ndarray
    terminal_during_opponent_internal: np.ndarray


class LearnerTurnEnv:
    """Fold opponent decisions until the learner is to act again or the env ends.

    The underlying batched env advances every unfinished row on each call to ``step``.
    Because of that, this wrapper can only keep folding while all unfinished rows still
    need opponent decisions together. If some rows return to learner-turn earlier than
    others, the wrapper raises instead of silently taking extra learner actions.
    """

    def __init__(
        self,
        env: DecisionBoundaryEnv,
        *,
        learner_seat: int,
        opponent_policy: OpponentPolicy,
        max_internal_steps: int = 512,
    ) -> None:
        if learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")
        if max_internal_steps < 1:
            raise ValueError("max_internal_steps must be >= 1")

        self._env = env
        self._learner_seat = int(learner_seat)
        self._opp_policy = opponent_policy
        self._cap = int(max_internal_steps)
        self._batch: Any | None = None

    def reset(self, seed: int | None = None) -> Any:
        batch = self._env.reset(seed=seed)
        self._batch = batch
        return batch

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    def step(self, learner_actions: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, LearnerTurnStepInfo]:
        if self._batch is None:
            raise RuntimeError("reset() must be called before step()")

        batch, reward_learn, done, info = self.step_from_batch(self._batch, learner_actions)
        self._batch = batch
        return batch, reward_learn, done, info

    def step_from_batch(
        self,
        batch: Any,
        learner_actions: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray, LearnerTurnStepInfo]:
        learner_actions_arr = np.asarray(learner_actions)
        if learner_actions_arr.ndim != 1:
            raise ValueError("learner_actions must be 1D [batch]")

        actor = self._get_actor(batch)
        done = self._get_done(batch)
        batch_size = actor.shape[0]
        if learner_actions_arr.shape != (batch_size,):
            raise ValueError("learner_actions length must match batch size")

        reward_learn = np.zeros((batch_size,), dtype=np.float32)
        k_raw = np.zeros((batch_size,), dtype=np.int32)
        terminal_during_opp = np.zeros((batch_size,), dtype=bool)
        current_batch = batch
        step_count = 0

        while True:
            active = ~done
            if not np.any(active):
                break
            if step_count >= self._cap:
                raise RuntimeError(f"LearnerTurnEnv safety cap exceeded ({self._cap})")

            actions = np.zeros((batch_size,), dtype=np.int64)
            learner_mask = active & (actor == self._learner_seat)
            opponent_mask = active & ~learner_mask

            if np.any(learner_mask):
                actions[learner_mask] = learner_actions_arr[learner_mask].astype(np.int64, copy=False)
            if np.any(opponent_mask):
                opponent_actions = np.asarray(self._opp_policy(current_batch, opponent_mask), dtype=np.int64)
                if opponent_actions.shape != (batch_size,):
                    raise ValueError("opponent_policy must return actions shaped [B]")
                actions[opponent_mask] = opponent_actions[opponent_mask]

            actor_before = actor.copy()
            done_before = done.copy()
            current_batch = self._env.step(actions)
            step_count += 1

            step_reward = self._get_reward(current_batch)
            sign = np.where(actor_before == self._learner_seat, 1.0, -1.0).astype(np.float32)
            reward_learn[active] += sign[active] * step_reward[active]
            k_raw[active] += 1

            done = self._get_done(current_batch)
            actor = self._get_actor(current_batch)
            newly_done = (~done_before) & done
            terminal_during_opp |= newly_done & (actor_before != self._learner_seat)

            unfinished = ~done
            if not np.any(unfinished):
                break

            ready_for_learner = unfinished & (actor == self._learner_seat)
            still_opponent = unfinished & ~ready_for_learner
            if np.any(ready_for_learner) and np.any(still_opponent):
                raise RuntimeError(
                    "LearnerTurnEnv cannot safely fold a diverged batch once some rows return to learner turn "
                    "earlier than others"
                )
            if np.any(ready_for_learner):
                break

        info = LearnerTurnStepInfo(
            k_raw_decisions=k_raw,
            terminal_during_opponent_internal=terminal_during_opp,
        )
        return current_batch, reward_learn, done, info

    @staticmethod
    def _get_actor(batch: Any) -> np.ndarray:
        for attr_name in ("actor", "to_play_seat"):
            actor = getattr(batch, attr_name, None)
            if actor is not None:
                arr = np.asarray(actor)
                if arr.ndim != 1:
                    raise ValueError(f"batch.{attr_name} must be 1D [B]")
                return arr
        raise AttributeError("batch must expose .actor or .to_play_seat [B]")

    @staticmethod
    def _get_reward(batch: Any) -> np.ndarray:
        for attr_name in ("reward", "rewards"):
            reward = getattr(batch, attr_name, None)
            if reward is not None:
                arr = np.asarray(reward, dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(f"batch.{attr_name} must be 1D [B]")
                return arr
        raise AttributeError("batch must expose .reward or .rewards [B]")

    @staticmethod
    def _get_done(batch: Any) -> np.ndarray:
        done = getattr(batch, "done", None)
        if done is not None:
            arr = np.asarray(done, dtype=bool)
            if arr.ndim != 1:
                raise ValueError("batch.done must be 1D [B]")
            return arr

        terminated = np.asarray(batch.terminated, dtype=bool)
        truncated = np.asarray(batch.truncated, dtype=bool)
        if terminated.ndim != 1:
            raise ValueError("batch.terminated must be 1D [B]")
        if truncated.ndim != 1:
            raise ValueError("batch.truncated must be 1D [B]")
        return terminated | truncated
