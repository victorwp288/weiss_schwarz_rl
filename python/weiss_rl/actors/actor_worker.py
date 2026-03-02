"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ActorWorker:
    actor_id: int

    def run_once(self) -> None:
        """Single iteration placeholder for rollout collection."""
        # TODO: connect env -> policy -> trajectory buffer pipeline.
        return None
