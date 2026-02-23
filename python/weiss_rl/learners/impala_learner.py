"""IMPALA learner scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ImpalaLearner:
    learning_rate: float = 2e-4


    def update(self, batch) -> dict[str, float]:
        """Placeholder learner update hook."""
        # TODO: connect v-trace loss and optimizer update.
        _ = batch
        return {"loss": 0.0}
