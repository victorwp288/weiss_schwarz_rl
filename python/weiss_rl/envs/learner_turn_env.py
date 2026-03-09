"""Optional learner-turn wrapper over decision-boundary steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LearnerTurnEnv:
    """Thin pass-through wrapper for a single learning-seat view."""

    base_env: Any
    learning_seat: int = 0

    def reset(self, seed: int | None = None) -> Any:
        return self.base_env.reset(seed=seed)

    def step(self, actions: Any) -> Any:
        return self.base_env.step(actions)
