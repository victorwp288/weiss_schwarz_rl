"""Decision-boundary environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import weiss_sim


@dataclass(slots=True)
class DecisionBoundaryEnv:
    """Thin wrapper around `weiss_sim.make` for policy-step interaction."""

    sim: Any


    @classmethod
    def create(cls, **kwargs: Any) -> "DecisionBoundaryEnv":
        sim = weiss_sim.make(**kwargs)
        return cls(sim=sim)


    def reset(self, seed: int | None = None):
        return self.sim.reset(seed=seed)


    def step(self, actions):
        return self.sim.step(actions)


    def close(self) -> None:
        close_fn = getattr(self.sim, "close", None)
        if callable(close_fn):
            close_fn()
