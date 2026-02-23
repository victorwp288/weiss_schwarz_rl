"""Payoff matrix helpers."""

from __future__ import annotations

import numpy as np


def to_antisymmetric(payoff: np.ndarray) -> np.ndarray:
    arr = np.asarray(payoff, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("payoff must be a square matrix")
    return 0.5 * (arr - arr.T)
