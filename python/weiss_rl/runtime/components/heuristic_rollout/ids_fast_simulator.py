"""Pooled simulator interface used by the heuristic ids-offset fast rollout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from weiss_rl.envs.decision_env import _pack_batch


@dataclass(frozen=True)
class HeuristicIdsFastSimulator:
    pool: Any
    step_out: Any
    step_into_i16_legal_ids: Any
    reset_done_into_i16_legal_ids: Any

    @classmethod
    def from_env(cls, env: Any) -> HeuristicIdsFastSimulator:
        pool = getattr(env, "pool", None)
        if pool is None:
            raise RuntimeError("heuristic ids fast path requires a pooled simulator env")
        step_into = getattr(pool, "step_into_i16_legal_ids", None)
        reset_done_into = getattr(pool, "reset_done_into_i16_legal_ids", None)
        if not callable(step_into) or not callable(reset_done_into):
            raise RuntimeError(
                "heuristic ids fast path requires pool.step_into_i16_legal_ids(...) "
                "and pool.reset_done_into_i16_legal_ids(...)"
            )
        step_out = getattr(env, "_step_out", None)
        if step_out is None:
            step_out = env._require_step_out(__import__("weiss_sim"))
        return cls(
            pool=pool,
            step_out=step_out,
            step_into_i16_legal_ids=step_into,
            reset_done_into_i16_legal_ids=reset_done_into,
        )

    def step(self, env: Any, actions: np.ndarray) -> None:
        self.step_into_i16_legal_ids(np.asarray(actions, dtype=np.uint32), self.step_out)
        env._handle_engine_status(self.step_out, weiss_sim=None)

    def reset_done(self, env: Any, done: np.ndarray) -> None:
        self.reset_done_into_i16_legal_ids(np.ascontiguousarray(done, dtype=np.bool_), self.step_out)
        env._handle_engine_status(self.step_out, weiss_sim=None)

    def pack_terminal_batch(self) -> Any:
        return _pack_batch(
            self.step_out,
            legality="ids_offsets",
            pool=self.pool,
            copy_arrays=True,
        )

    def pack_next_batch(self, runtime: Any) -> Any:
        next_batch = _pack_batch(
            self.step_out,
            legality="ids_offsets",
            pool=self.pool,
            copy_arrays=False,
        )
        if next_batch.ids_offsets is None or next_batch.legal_action_meta is not None:
            return next_batch
        legal_meta_builder = getattr(runtime, "_legal_action_meta_from_ids", None)
        next_legal_action_meta = legal_meta_builder(next_batch.ids_offsets[0]) if callable(legal_meta_builder) else None
        if next_legal_action_meta is None:
            return next_batch
        return replace(next_batch, legal_action_meta=next_legal_action_meta)


__all__ = ["HeuristicIdsFastSimulator"]
