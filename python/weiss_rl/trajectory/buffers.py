"""Simple in-memory trajectory buffers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import TrajectoryStep


@dataclass(slots=True)
class TrajectoryBuffer:
    """Append-only buffer used by actor workers before learner handoff."""

    steps: list[TrajectoryStep] = field(default_factory=list)


    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)


    def clear(self) -> None:
        self.steps.clear()
