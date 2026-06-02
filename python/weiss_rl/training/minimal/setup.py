"""Compatibility surface for minimal training setup."""

from __future__ import annotations

from weiss_rl.training.loop.setup import (
    MinimalTrainingSetup as MinimalTrainingSetup,
)
from weiss_rl.training.loop.setup import (
    MinimalTrainingSetupHooks as MinimalTrainingSetupHooks,
)
from weiss_rl.training.loop.setup import (
    build_minimal_training_setup as build_minimal_training_setup,
)
from weiss_rl.training.loop.setup import (
    require_training_stack_components as _require_training_stack_components,
)

__all__ = [
    "MinimalTrainingSetup",
    "MinimalTrainingSetupHooks",
    "_require_training_stack_components",
    "build_minimal_training_setup",
]
