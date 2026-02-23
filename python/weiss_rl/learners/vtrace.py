"""V-trace helper scaffold."""

from __future__ import annotations

import numpy as np


def compute_vtrace_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    discounts: np.ndarray,
) -> np.ndarray:
    """Placeholder V-trace target calculation.

    This simple bootstrap is a scaffold only; replace with full IMPALA V-trace.
    """
    if rewards.shape != values.shape or rewards.shape != discounts.shape:
        raise ValueError("rewards, values, discounts must have identical shapes")
    return rewards + discounts * values
