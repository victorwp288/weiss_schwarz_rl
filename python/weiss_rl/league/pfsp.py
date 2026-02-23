"""Prioritized Fictitious Self-Play utilities."""

from __future__ import annotations

import numpy as np


def pfsp_probabilities(win_rates: np.ndarray, power: float = 2.0, eps_uniform: float = 0.2) -> np.ndarray:
    """Compute PFSP sampling distribution from win-rate estimates."""
    wr = np.asarray(win_rates, dtype=np.float64)
    if wr.ndim != 1:
        raise ValueError("win_rates must be 1D")
    base = np.power(np.clip(1.0 - wr, 0.0, 1.0), power)
    if not np.any(base > 0):
        base = np.ones_like(base)
    base = base / np.sum(base)
    uniform = np.full_like(base, 1.0 / base.size)
    return (1.0 - eps_uniform) * base + eps_uniform * uniform
