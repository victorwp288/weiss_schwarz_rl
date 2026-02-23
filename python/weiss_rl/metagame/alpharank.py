"""AlphaRank scaffold."""

from __future__ import annotations

import numpy as np


def normalize_stationary(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("scores must be non-negative")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("sum(scores) must be > 0")
    return arr / total
