"""Zero-sum Nash mixture solver helpers."""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

_DEFAULT_LP_BACKEND = "scipy_linprog_highs"

__all__ = [
    "NashSolveResult",
    "NashSolverReport",
    "solve_nash_mixture",
    "solve_zero_sum_mixture",
    "uniform_mixture",
    "write_mixture_mean_csv",
    "write_nash_artifacts",
    "write_solver_report_json",
]


def _run_highs_linprog(*args: Any, threads: int | None = None, **kwargs: Any):
    options = dict(kwargs.pop("options", {}))
    if threads is not None:
        options["threads"] = int(threads)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: \{'threads': .*",
            category=OptimizeWarning,
        )
        return linprog(*args, method="highs", options=options, **kwargs)


@dataclass(frozen=True, slots=True)
class NashSolveResult:
    mixture: np.ndarray
    value: float
    solver_status: int
    solver_message: str


@dataclass(frozen=True, slots=True)
class NashSolverReport:
    solver: str
    backend: str
    status: int
    success: bool
    message: str
    value: float
    actual_game_value: float
    mixture: tuple[float, ...]
    policy_ids: tuple[str, ...] | None
    threads: int
    tie_break: str
    value_tolerance: float
    bias_scale: float
    max_inequality_violation: float
    max_equality_violation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "backend": self.backend,
            "status": self.status,
            "success": self.success,
            "message": self.message,
            "value": self.value,
            "actual_game_value": self.actual_game_value,
            "mixture": list(self.mixture),
            "policy_ids": list(self.policy_ids) if self.policy_ids is not None else None,
            "threads": self.threads,
            "tie_break": self.tie_break,
            "value_tolerance": self.value_tolerance,
            "bias_scale": self.bias_scale,
            "max_inequality_violation": self.max_inequality_violation,
            "max_equality_violation": self.max_equality_violation,
        }


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


def solve_nash_mixture(
    p_mean: np.ndarray,
    *,
    policy_ids: Sequence[str] | None = None,
    value_tolerance: float = 1e-9,
    tie_break: str = "lowest_policy_id",
    threads: int = 1,
) -> tuple[np.ndarray, NashSolverReport]:
    """Solve the symmetric zero-sum Nash mixture for a payoff mean matrix."""

    p_mean_arr = np.asarray(p_mean, dtype=np.float64)
    normalized_policy_ids = None if policy_ids is None else tuple(str(policy_id) for policy_id in policy_ids)
    normalized_value_tolerance = _validate_value_tolerance(value_tolerance)
    normalized_threads = _validate_threads(threads)
    _validate_probability_payoff_matrix(p_mean_arr, value_tolerance=normalized_value_tolerance)

    policy_count = p_mean_arr.shape[0]
    if policy_count == 0:
        raise ValueError("p_mean must contain at least one policy")
    if normalized_policy_ids is not None and len(normalized_policy_ids) != policy_count:
        raise ValueError("policy_ids length must match p_mean dimensions")
    if normalized_policy_ids is not None and len(set(normalized_policy_ids)) != len(normalized_policy_ids):
        raise ValueError("policy_ids must be unique")
    if tie_break not in {"lowest_policy_id", "policy_index"}:
        raise ValueError(f"unsupported tie_break: {tie_break!r}")
    if tie_break == "lowest_policy_id" and normalized_policy_ids is None:
        raise ValueError("policy_ids must be provided for lowest_policy_id tie-break")

    if policy_count == 1:
        mixture = np.asarray([1.0], dtype=np.float64)
        value = float(p_mean_arr[0, 0])
        report = NashSolverReport(
            solver="linprog",
            backend="highs",
            status=0,
            success=True,
            message="single_policy",
            value=value,
            actual_game_value=value,
            mixture=(1.0,),
            policy_ids=normalized_policy_ids,
            threads=normalized_threads,
            tie_break=tie_break,
            value_tolerance=normalized_value_tolerance,
            bias_scale=1.0,
            max_inequality_violation=0.0,
            max_equality_violation=0.0,
        )
        return mixture, report

    if tie_break == "lowest_policy_id":
        assert normalized_policy_ids is not None
        lexical_order = np.argsort(np.asarray(normalized_policy_ids, dtype=object), kind="stable")
        inverse_rank = np.empty(policy_count, dtype=np.int64)
        inverse_rank[lexical_order] = np.arange(policy_count, dtype=np.int64)
        bias = inverse_rank.astype(np.float64)
    else:
        bias = np.arange(policy_count, dtype=np.float64)

    c_primary = np.concatenate([np.zeros((policy_count,), dtype=np.float64), np.array([-1.0], dtype=np.float64)])
    a_ub = np.concatenate([-p_mean_arr.T, np.ones((policy_count, 1), dtype=np.float64)], axis=1)
    b_ub = np.zeros((policy_count,), dtype=np.float64)
    a_eq = np.concatenate([np.ones((1, policy_count), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)], axis=1)
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, None)] * policy_count + [(None, None)]

    primary_result = _run_highs_linprog(
        c_primary,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        threads=normalized_threads,
    )
    if not primary_result.success or primary_result.x is None:
        raise ValueError(
            "Nash solver failed: "
            + str(primary_result.message)
            + f" (status={primary_result.status}, thread_count={normalized_threads})"
        )

    primary_solution = np.asarray(primary_result.x, dtype=np.float64)
    value = float(primary_solution[policy_count])

    c_secondary = bias
    a_ub_secondary = -p_mean_arr.T
    b_ub_secondary = np.full((policy_count,), -(value - normalized_value_tolerance), dtype=np.float64)
    a_eq_secondary = np.ones((1, policy_count), dtype=np.float64)
    b_eq_secondary = np.array([1.0], dtype=np.float64)
    bounds_secondary = [(0.0, None)] * policy_count
    secondary_result = _run_highs_linprog(
        c_secondary,
        A_ub=a_ub_secondary,
        b_ub=b_ub_secondary,
        A_eq=a_eq_secondary,
        b_eq=b_eq_secondary,
        bounds=bounds_secondary,
        threads=normalized_threads,
    )
    if not secondary_result.success or secondary_result.x is None:
        raise ValueError(
            "Nash tie-break LP failed: "
            + str(secondary_result.message)
            + f" (status={secondary_result.status}, thread_count={normalized_threads})"
        )

    solution = np.asarray(secondary_result.x, dtype=np.float64)
    max_ineq = float(np.max(np.maximum(0.0, a_ub_secondary.dot(solution) - b_ub_secondary)))
    max_eq = float(np.max(np.abs(a_eq_secondary.dot(solution) - b_eq_secondary)))

    mixture = np.clip(solution, 0.0, None)
    mixture_sum = float(np.sum(mixture))
    if mixture_sum <= 0.0:
        raise ValueError("Nash solver produced a non-positive mixture")
    mixture /= mixture_sum

    expected_values = p_mean_arr.T.dot(mixture)
    actual_game_value = float(np.min(expected_values))
    report = NashSolverReport(
        solver="linprog",
        backend="highs",
        status=int(primary_result.status),
        success=bool(primary_result.success),
        message=str(primary_result.message),
        value=value,
        actual_game_value=actual_game_value,
        mixture=tuple(float(x) for x in mixture.tolist()),
        policy_ids=normalized_policy_ids,
        threads=normalized_threads,
        tie_break=tie_break,
        value_tolerance=normalized_value_tolerance,
        bias_scale=1.0,
        max_inequality_violation=max_ineq,
        max_equality_violation=max_eq,
    )
    return mixture, report


def write_mixture_mean_csv(path: Path, policy_ids: Sequence[str], mixture: np.ndarray) -> None:
    """Write a Nash equilibrium mixture to a CSV file."""

    mixture_arr = np.asarray(mixture, dtype=np.float64)
    if mixture_arr.ndim != 1:
        raise ValueError("mixture must be a one-dimensional array")
    if len(policy_ids) != mixture_arr.shape[0]:
        raise ValueError("policy_ids length must match mixture length")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy_id", "mixture"])
        for policy_id, weight in zip(policy_ids, mixture_arr.tolist(), strict=True):
            writer.writerow([policy_id, f"{float(weight):.12g}"])


def write_solver_report_json(path: Path, report: NashSolverReport) -> None:
    """Write a Nash solver report to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_nash_artifacts(
    mixture_mean_csv: Path,
    solver_report_json: Path,
    mixture: np.ndarray,
    report: NashSolverReport,
    policy_ids: Sequence[str],
) -> None:
    write_mixture_mean_csv(mixture_mean_csv, policy_ids, mixture)
    write_solver_report_json(solver_report_json, report)


def _validate_payoff(payoff: np.ndarray) -> np.ndarray:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("payoff must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("payoff must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff must be finite")
    return matrix


def _validate_probability_payoff_matrix(p_mean: np.ndarray, *, value_tolerance: float) -> None:
    if p_mean.ndim != 2 or p_mean.shape[0] != p_mean.shape[1]:
        raise ValueError("p_mean must be a square matrix")
    if not np.isfinite(p_mean).all():
        raise ValueError("p_mean must contain only finite values")
    if np.any((p_mean < 0.0) | (p_mean > 1.0)):
        raise ValueError("p_mean entries must lie within [0, 1]")
    if p_mean.size == 0:
        return
    if not np.allclose(np.diag(p_mean), 0.5, atol=value_tolerance, rtol=0.0):
        raise ValueError("p_mean diagonal must be 0.5")

    off_diagonal = ~np.eye(p_mean.shape[0], dtype=bool)
    reciprocal_sums = p_mean + p_mean.T
    if not np.allclose(reciprocal_sums[off_diagonal], 1.0, atol=value_tolerance, rtol=0.0):
        raise ValueError("p_mean off-diagonal entries must be reciprocal and sum to 1")


def _validate_threads(threads: int) -> int:
    if isinstance(threads, bool):
        raise ValueError("threads must be a positive finite integer")
    try:
        numeric = float(threads)
    except (TypeError, ValueError) as exc:
        raise ValueError("threads must be a positive finite integer") from exc
    if not np.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError("threads must be a positive finite integer")
    return int(numeric)


def _validate_value_tolerance(value_tolerance: float) -> float:
    normalized = float(value_tolerance)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError("value_tolerance must be nonnegative and finite")
    return normalized


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
