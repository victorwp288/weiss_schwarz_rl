"""Optional learner-turn wrapper over decision-boundary steps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LearnerTurnEnv:
    """Scaffold wrapper for future learner-seat turn folding logic."""

    base_env: object
    learning_seat: int = 0


    def reset(self, seed: int | None = None):
        return self.base_env.reset(seed=seed)


    def step(self, actions):
        # TODO: fold opponent-internal decisions per masterplan §5.2.
        return self.base_env.step(actions)
