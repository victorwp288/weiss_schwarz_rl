"""Zero-sum Nash mixture solver helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

_DEFAULT_LP_BACKEND = "scipy_linprog_highs"

__all__ = [
    "NashSolveResult",
    "solve_zero_sum_mixture",
    "uniform_mixture",
]


@dataclass(frozen=True, slots=True)
class NashSolveResult:
    mixture: np.ndarray
    value: float
    solver_status: int
    solver_message: str


def uniform_mixture(num_policies: int) -> np.ndarray:
    if num_policies <= 0:
        raise ValueError("num_policies must be > 0")
    return np.full((num_policies,), 1.0 / num_policies, dtype=np.float64)


def solve_zero_sum_mixture(
    payoff: np.ndarray,
    *,
    policy_ids: Sequence[str] | None = None,
    backend: str = _DEFAULT_LP_BACKEND,
    value_tolerance: float = 1.0e-10,
) -> NashSolveResult:
    """Solve a row-player zero-sum equilibrium on a square payoff matrix."""

    matrix = _validate_payoff(payoff)
    policy_count = matrix.shape[0]
    if policy_count == 1:
        return NashSolveResult(
            mixture=np.asarray([1.0], dtype=np.float64),
            value=float(matrix[0, 0]),
            solver_status=0,
            solver_message="single_policy",
        )
    if backend != _DEFAULT_LP_BACKEND:
        raise ValueError(f"unsupported Nash backend: {backend!r}")

    primary = _solve_primary_lp(matrix)
    if not primary.success or primary.x is None:
        raise RuntimeError(f"Nash LP failed: status={primary.status} message={primary.message}")
    primary_mixture = _normalize_probability_vector(primary.x[:-1])
    value = float(primary.x[-1])

    secondary = _solve_tie_break_lp(
        matrix,
        policy_ids=policy_ids,
        value=value,
        value_tolerance=value_tolerance,
    )
    mixture = primary_mixture
    if secondary.success and secondary.x is not None:
        mixture = _normalize_probability_vector(secondary.x)

    return NashSolveResult(
        mixture=mixture,
        value=value,
        solver_status=int(primary.status),
        solver_message=str(primary.message),
    )


def _validate_payoff(payoff: np.ndarray) -> np.ndarray:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("payoff must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("payoff must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff must be finite")
    return matrix


def _solve_primary_lp(matrix: np.ndarray):
    policy_count = matrix.shape[0]
    objective = np.zeros((policy_count + 1,), dtype=np.float64)
    objective[-1] = -1.0
    inequality = np.hstack((-matrix.T, np.ones((policy_count, 1), dtype=np.float64)))
    inequality_rhs = np.zeros((policy_count,), dtype=np.float64)
    equality = np.zeros((1, policy_count + 1), dtype=np.float64)
    equality[0, :policy_count] = 1.0
    equality_rhs = np.asarray([1.0], dtype=np.float64)
    bounds = [(0.0, None) for _ in range(policy_count)] + [(None, None)]
    return linprog(
        c=objective,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )


def _solve_tie_break_lp(
    matrix: np.ndarray,
    *,
    policy_ids: Sequence[str] | None,
    value: float,
    value_tolerance: float,
):
    policy_count = matrix.shape[0]
    objective = -_tie_break_weights(policy_count, policy_ids=policy_ids)
    inequality = -matrix.T
    inequality_rhs = np.full((policy_count,), -(value - value_tolerance), dtype=np.float64)
    equality = np.ones((1, policy_count), dtype=np.float64)
    equality_rhs = np.asarray([1.0], dtype=np.float64)
    bounds = [(0.0, None) for _ in range(policy_count)]
    return linprog(
        c=objective,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )


def _tie_break_weights(policy_count: int, *, policy_ids: Sequence[str] | None) -> np.ndarray:
    if policy_ids is None:
        order = list(range(policy_count))
    else:
        if len(policy_ids) != policy_count:
            raise ValueError("policy_ids length must match payoff shape")
        order = sorted(range(policy_count), key=lambda index: str(policy_ids[index]))
    weights = np.zeros((policy_count,), dtype=np.float64)
    scale = 1.0
    for index in reversed(order):
        weights[index] = scale
        scale *= 2.0
    return weights


def _normalize_probability_vector(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0.0:
        raise RuntimeError("Nash solver produced a zero-mass mixture")
    normalized = clipped / total
    normalized[np.abs(normalized) < 1.0e-12] = 0.0
    return normalized
