"""All-heuristic actor rollout fast paths for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from weiss_rl.runtime.components.heuristic_rollout.ids_fast import QueueRuntimeHeuristicIdsFastRolloutMixin
from weiss_rl.runtime.components.heuristic_rollout.native import QueueRuntimeHeuristicNativeRolloutMixin


class QueueRuntimeHeuristicRolloutMixin(
    QueueRuntimeHeuristicNativeRolloutMixin,
    QueueRuntimeHeuristicIdsFastRolloutMixin,
):
    """Compose all-heuristic actor rollout implementations."""


__all__ = ["QueueRuntimeHeuristicRolloutMixin"]
