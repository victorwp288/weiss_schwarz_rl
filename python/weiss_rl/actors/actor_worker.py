"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ActorWorker:
    actor_id: int

    def run_once(self) -> None:
        """Single rollout-collection hook for the actor loop."""
        return None
