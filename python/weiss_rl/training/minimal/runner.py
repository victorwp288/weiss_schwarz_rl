"""Compatibility surface for minimal training update-loop execution."""

from __future__ import annotations

from weiss_rl.training.loop.runner import (
    MinimalTrainingRunHooks as MinimalTrainingRunHooks,
)
from weiss_rl.training.loop.runner import (
    run_minimal_training_updates as run_minimal_training_updates,
)

__all__ = [
    "MinimalTrainingRunHooks",
    "run_minimal_training_updates",
]
