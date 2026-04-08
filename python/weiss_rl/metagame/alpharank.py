"""AlphaRank implementation for policy ranking in metagames."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def normalize_stationary(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("scores must be non-negative")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("sum(scores) must be > 0")
    return arr / total


def compute_alpharank_stationary(
    p_mean: np.ndarray,
    *,
    policy_ids: Sequence[str] | None = None,
    m: int = 50,
    alpha: int = 100,
    local_selection: bool = True,
    use_inf_alpha: bool = False,
    inf_alpha_eps: float = 0.01,
) -> np.ndarray:
    """Compute the AlphaRank stationary distribution for policy ranking.

    Args:
        p_mean: Payoff mean matrix (n x n).
        policy_ids: Optional policy identifiers.
        m: Number of iterations.
        alpha: Selection strength parameter.
        local_selection: Whether to use local selection model.
        use_inf_alpha: Whether to use infinite alpha approximation.
        inf_alpha_eps: Epsilon for infinite alpha approximation.

    Returns:
        Stationary distribution as numpy array.
    """
    p_mean_arr = np.asarray(p_mean, dtype=np.float64)
    if p_mean_arr.ndim != 2 or p_mean_arr.shape[0] != p_mean_arr.shape[1]:
        raise ValueError("p_mean must be a square matrix")

    n = p_mean_arr.shape[0]
    if n == 0:
        raise ValueError("p_mean must contain at least one policy")
    if policy_ids is not None and len(policy_ids) != n:
        raise ValueError("policy_ids length must match p_mean dimensions")
    if np.isnan(p_mean_arr).any():
        raise ValueError("p_mean must not contain NaN values")
    if not local_selection:
        raise NotImplementedError("Global selection not implemented")

    # Initialize uniform distribution
    pi = np.ones(n, dtype=np.float64) / n

    for _ in range(m):
        # Compute fitness: expected payoff against current population
        fitness = p_mean_arr @ pi

        # Compute transition probabilities P[j,i] = prob that i beats j
        diff = fitness[:, None] - fitness[None, :]
        
        if use_inf_alpha:
            # Infinite alpha: i beats j if fitness_i > fitness_j - eps
            P = (diff > -inf_alpha_eps).astype(np.float64)
        else:
            # Sigmoid with selection strength alpha
            P = 1.0 / (1.0 + np.exp(-alpha * diff))

        # Update distribution: π_{t+1} = π_t @ P^T
        pi = pi @ P.T
        
        # Normalize to ensure it's a valid distribution
        pi = normalize_stationary(pi)

    return pi


def write_stationary_mean_csv(
    path: Path, 
    policy_ids: Sequence[str], 
    stationary: np.ndarray
) -> None:
    """Write the AlphaRank stationary distribution to a CSV file."""
    stationary_arr = np.asarray(stationary, dtype=np.float64)
    if stationary_arr.ndim != 1:
        raise ValueError("stationary must be a one-dimensional array")
    if len(policy_ids) != stationary_arr.shape[0]:
        raise ValueError("policy_ids length must match stationary length")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy_id", "stationary_probability"])
        for policy_id, prob in zip(policy_ids, stationary_arr.tolist()):
            writer.writerow([policy_id, f"{float(prob):.12g}"])


def write_alpharank_artifacts(
    stationary_mean_csv: Path,
    stationary: np.ndarray,
    policy_ids: Sequence[str],
) -> None:
    """Write AlphaRank artifacts to files."""
    write_stationary_mean_csv(stationary_mean_csv, policy_ids, stationary)
