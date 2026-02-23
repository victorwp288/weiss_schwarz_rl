"""Uncertainty estimation helpers."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return mean, lo, hi
