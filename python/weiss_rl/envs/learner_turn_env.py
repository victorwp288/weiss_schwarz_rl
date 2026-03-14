"""Optional learner-turn wrapper over decision-boundary steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Minimal protocol for a decision-boundary batched env.
class DecisionBoundaryEnv(Protocol):
    def reset(self) -> Any: ...
    def step(self, actions: np.ndarray) -> Any: ...


# OpponentPolicy(batch, opponent_mask: bool[B]) -> actions int64[B]
OpponentPolicy = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(slots=True)
class LearnerTurnStepInfo:
    k_raw_decisions: np.ndarray #int32 [B]
    terminal_during_opponent_internal: np.ndarray # bool [B]

"""Fold opponent internal decisions so one external step equals one learner decision.

    This wrapper is optional and intended for specific ablations.
    It is slower than direct DecisionBoundary stepping because it may loop in Python.
"""

class LearnerTurnEnv:
    def __init__(
        self,
        env: DecisionBoundaryEnv,
        *,
        learner_seat: int,
        opponent_policiy: OpponentPolicy,
        max_internal_steps: int = 512,        
    ) -> None:
        if learner_seat not in (0, 1):
            raise ValueError("learner_set must be 0 or 1")
        if max_internal_steps < 1:
            raise ValueError("max_internal_steps must be >=1")
        
        self._env = env
        self._learner_seat = int(learner_seat)
        self._opp_policy = opponent_policiy
        self._cap = int(max_internal_steps)
    
    def reset(self) -> Any:
        return self._env.reset()
    
    """Perform one learner-turn step.

        Args:
            learner_actions: int array [B], only used where actor == learner_seat and not done.
                            For other rows, ignored.

        Returns:
            batch: underlying batch after folding (at learner turn boundary or terminal)
            reward_learn: float32 [B], learner-perspective aggregated reward
            done: bool [B]
            info: LearnerTurnStepInfo
    """
    def step(self, learner_actions: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, LearnerTurnStepInfo]:
        learner_actions = np.asarray(learner_actions)
        if learner_actions.ndim != 1:
            raise ValueError("learner_actions must be 1D [batch]")
        
        raise learner_actions = np.asarray(learner_actions)
        if learner_actions.ndim != 1:
            raise ValueError("learner_actions must be 1D [batch]")

    def step_from_batch(
        self,
        batch: Any,
        learner_actions: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray, LearnerTurnStepInfo]:
        """Same as step(), but explicit about the current batch."""
        learner_actions = np.asarray(learner_actions)
        if learner_actions.ndim != 1:
            raise ValueError("learner_actions must be 1D [batch]")

        actor = self._get_actor(batch)                 # int [B]
        done = self._get_done(batch)                   # bool [B]

        bsz = actor.shape[0]
        if learner_actions.shape[0] != bsz:
            raise ValueError("learner_actions length must match batch size")

        k_raw = np.zeros((bsz,), dtype=np.int32)
        terminal_during_opp = np.zeros((bsz,), dtype=bool)
        reward_learn = np.zeros((bsz,), dtype=np.float32)

        # Helper: apply reward in learner perspective per step:
        # simulator reward is actor perspective at that decision boundary.
        def _accumulate(step_batch: Any) -> None:
            step_actor = self._get_actor(step_batch)
            step_reward = self._get_reward(step_batch).astype(np.float32, copy=False)
            # learner perspective: +r when learner acted, -r when opponent acted
            sign = np.where(step_actor == self._learner_seat, 1.0, -1.0).astype(np.float32)
            reward_learn[:] += sign * step_reward

        # 1) learner decision step for rows where learner to act and not done
        need_learner = (~done) & (actor == self._learner_seat)
        if np.any(need_learner):
            actions = np.zeros((bsz,), dtype=np.int64)
            actions[need_learner] = learner_actions[need_learner].astype(np.int64, copy=False)
            batch = self._env.step(actions)
            k_raw[need_learner] += 1
            _accumulate(batch)
            done = self._get_done(batch)
            actor = self._get_actor(batch)

        # If learner was not to act for some envs, we still must count at least 1 raw decision
        # for those envs once we execute the first internal opponent step.
        # We now fold opponent turns until learner turn or terminal.
        internal_steps = 0
        while True:
            if internal_steps >= self._cap:
                raise RuntimeError(f"LearnerTurnEnv safety cap exceeded ({self._cap})")

            # Stop if all active envs are at learner turn or terminal.
            active = ~done
            need_continue = active & (actor != self._learner_seat)
            if not np.any(need_continue):
                break

            # Ask opponent policy for actions for opponent rows only.
            opp_actions = self._opp_policy(batch, need_continue)
            opp_actions = np.asarray(opp_actions, dtype=np.int64)
            if opp_actions.shape != (bsz,):
                raise ValueError("opponent_policy must return actions shaped [B]")

            batch = self._env.step(opp_actions)
            # Count raw decisions only for rows that we stepped.
            k_raw[need_continue] += 1
            _accumulate(batch)

            prev_done = done
            done = self._get_done(batch)
            actor = self._get_actor(batch)

            # Terminal occurred during opponent internal decisions:
            newly_done = (~prev_done) & done
            terminal_during_opp |= newly_done & need_continue

            internal_steps += 1

        # Contract: k_raw_decisions >= 1 for any env that was not already done.
        # If an env starts done (should not happen for sane loops), leave 0.
        k_raw = np.where((~self._get_done(batch)) | (reward_learn != 0.0) | (k_raw > 0), np.maximum(k_raw, 1), k_raw)

        info = LearnerTurnStepInfo(
            k_raw_decisions=k_raw,
            terminal_during_opponent_internal=terminal_during_opp,
        )
        return batch, reward_learn, done, info

    @staticmethod
    def _get_actor(batch: Any) -> np.ndarray:
        actor = getattr(batch, "actor", None)
        if actor is None:
            raise AttributeError("batch must expose .actor [B]")
        return np.asarray(actor)

    @staticmethod
    def _get_reward(batch: Any) -> np.ndarray:
        rewards = getattr(batch, "rewards", None)
        if rewards is None:
            raise AttributeError("batch must expose .rewards [B]")
        return np.asarray(rewards)

    @staticmethod
    def _get_done(batch: Any) -> np.ndarray:
        terminated = np.asarray(getattr(batch, "terminated"))
        truncated = np.asarray(getattr(batch, "truncated"))
        return (terminated != 0) | (truncated != 0)
               