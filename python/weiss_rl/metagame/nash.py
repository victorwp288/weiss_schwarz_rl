"""Nash-mixture solver utilities."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import OptimizeWarning, linprog


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
    _validate_payoff_matrix(p_mean_arr, value_tolerance=normalized_value_tolerance)

    n = p_mean_arr.shape[0]
    if n == 0:
        raise ValueError("p_mean must contain at least one policy")
    if normalized_policy_ids is not None and len(normalized_policy_ids) != n:
        raise ValueError("policy_ids length must match p_mean dimensions")
    if normalized_policy_ids is not None and len(set(normalized_policy_ids)) != len(normalized_policy_ids):
        raise ValueError("policy_ids must be unique")
    if tie_break not in {"lowest_policy_id", "policy_index"}:
        raise ValueError(f"unsupported tie_break: {tie_break!r}")
    if tie_break == "lowest_policy_id" and normalized_policy_ids is None:
        raise ValueError("policy_ids must be provided for lowest_policy_id tie-break")

    if tie_break == "lowest_policy_id":
        if normalized_policy_ids is None:
            raise RuntimeError("policy_ids unexpectedly None for lowest_policy_id tie_break")
        lexical_order = np.argsort(np.asarray(normalized_policy_ids, dtype=object), kind="stable")
        inverse_rank = np.empty(n, dtype=np.int64)
        inverse_rank[lexical_order] = np.arange(n, dtype=np.int64)
        bias = inverse_rank.astype(np.float64)
    else:
        bias = np.arange(n, dtype=np.float64)

    bias_scale = 1.0
    c_primary = np.concatenate([np.zeros((n,), dtype=np.float64), np.array([-1.0], dtype=np.float64)])

    A_ub = np.concatenate([-p_mean_arr.T, np.ones((n, 1), dtype=np.float64)], axis=1)
    b_ub = np.zeros((n,), dtype=np.float64)
    A_eq = np.concatenate([np.ones((1, n), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)], axis=1)
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, None)] * n + [(None, None)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        primary_result = linprog(
            c_primary,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"threads": normalized_threads},
        )

    if not primary_result.success:
        raise ValueError(
            "Nash solver failed: "
            + primary_result.message
            + f" (status={primary_result.status}, thread_count={normalized_threads})"
        )

    primary_solution = np.asarray(primary_result.x, dtype=np.float64)
    value = float(primary_solution[n])

    if tie_break in {"lowest_policy_id", "policy_index"}:
        c_secondary = bias
        A_ub_secondary = -p_mean_arr.T
        b_ub_secondary = np.full((n,), -(value - normalized_value_tolerance), dtype=np.float64)
        A_eq_secondary = np.ones((1, n), dtype=np.float64)
        b_eq_secondary = np.array([1.0], dtype=np.float64)
        bounds_secondary = [(0.0, None)] * n

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning)
            secondary_result = linprog(
                c_secondary,
                A_ub=A_ub_secondary,
                b_ub=b_ub_secondary,
                A_eq=A_eq_secondary,
                b_eq=b_eq_secondary,
                bounds=bounds_secondary,
                method="highs",
                options={"threads": normalized_threads},
            )

        if not secondary_result.success:
            raise ValueError(
                "Nash tie-break LP failed: "
                + secondary_result.message
                + f" (status={secondary_result.status}, thread_count={normalized_threads})"
            )

        solution = np.asarray(secondary_result.x, dtype=np.float64)
        max_ineq = float(np.max(np.maximum(0.0, A_ub_secondary.dot(solution) - b_ub_secondary)))
        max_eq = float(np.max(np.abs(A_eq_secondary.dot(solution) - b_eq_secondary)))
    else:
        solution = primary_solution[:n]
        max_ineq = float(np.max(np.maximum(0.0, -A_ub.dot(primary_solution) + b_ub)))
        max_eq = float(np.max(np.abs(A_eq.dot(primary_solution) - b_eq)))

    mixture = np.clip(solution[:n], 0.0, None)
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
        bias_scale=float(bias_scale),
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


def _validate_payoff_matrix(p_mean: np.ndarray, *, value_tolerance: float) -> None:
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
