"""IMPALA learner scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImpalaLearner:
    learning_rate: float = 2e-4

    def update(self, batch: Any) -> dict[str, float]:
        """Learner update hook."""
        _ = batch
        return {"loss": 0.0}
