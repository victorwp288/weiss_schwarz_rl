"""AlphaRank implementation for policy ranking in metagames."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AlphaRankResult",
    "compute_alpharank_stationary",
    "compute_stationary_distribution",
    "normalize_stationary",
    "write_alpharank_artifacts",
    "write_stationary_mean_csv",
]

_STATIONARY_TOL = 1e-15
_STATIONARY_MAX_ITERS = 100_000
_TRANSITION_ATOL = 1e-12


@dataclass(frozen=True, slots=True)
class AlphaRankResult:
    stationary: np.ndarray
    transition_matrix: np.ndarray


def normalize_stationary(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("scores must be a non-empty vector")
    if np.any(arr < 0):
        raise ValueError("scores must be non-negative")
    total = float(np.sum(arr, dtype=np.float64))
    if total <= 0:
        raise ValueError("sum(scores) must be > 0")
    return arr / total


def compute_alpharank_stationary(
    p_mean: np.ndarray,
    *,
    policy_ids: Sequence[str] | None = None,
    m: int = 50,
    alpha: float = 100,
    local_selection: bool = True,
    use_inf_alpha: bool = False,
    inf_alpha_eps: float = 0.01,
) -> np.ndarray:
    """Compute the single-population AlphaRank stationary distribution."""
    return compute_stationary_distribution(
        p_mean,
        policy_ids=policy_ids,
        m=m,
        alpha=alpha,
        local_selection=local_selection,
        use_inf_alpha=use_inf_alpha,
        inf_alpha_eps=inf_alpha_eps,
    ).stationary


def compute_stationary_distribution(
    p_mean: np.ndarray,
    *,
    policy_ids: Sequence[str] | None = None,
    m: int = 50,
    alpha: float = 100,
    local_selection: bool = True,
    use_inf_alpha: bool = False,
    inf_alpha_eps: float = 0.01,
) -> AlphaRankResult:
    p_mean_arr = np.asarray(p_mean, dtype=np.float64)
    if p_mean_arr.ndim != 2 or p_mean_arr.shape[0] != p_mean_arr.shape[1]:
        raise ValueError("p_mean must be a square matrix")

    n = p_mean_arr.shape[0]
    if n == 0:
        raise ValueError("p_mean must contain at least one policy")
    if policy_ids is not None and len(policy_ids) != n:
        raise ValueError("policy_ids length must match p_mean dimensions")
    if not np.isfinite(p_mean_arr).all():
        raise ValueError("p_mean must contain only finite values")
    if m < 1:
        raise ValueError("m must be >= 1")
    if alpha < 1:
        raise ValueError("alpha must be >= 1")
    if inf_alpha_eps < 0.0:
        raise ValueError("inf_alpha_eps must be >= 0")
    if n == 1:
        stationary = np.ones(1, dtype=np.float64)
        return AlphaRankResult(stationary=stationary, transition_matrix=np.asarray([[1.0]], dtype=np.float64))

    transition_matrix = (
        _build_transition_matrix_local(
            p_mean_arr,
            m=m,
            alpha=alpha,
            use_inf_alpha=use_inf_alpha,
            tie_tolerance=inf_alpha_eps,
        )
        if local_selection
        else _build_transition_matrix_global(
            p_mean_arr,
            m=m,
            alpha=alpha,
            use_inf_alpha=use_inf_alpha,
            tie_tolerance=inf_alpha_eps,
        )
    )
    stationary = _solve_stationary_distribution(transition_matrix)
    return AlphaRankResult(stationary=stationary, transition_matrix=transition_matrix)


def write_stationary_mean_csv(path: Path, policy_ids: Sequence[str], stationary: np.ndarray) -> None:
    """Write the AlphaRank stationary distribution to a ranked CSV file."""
    stationary_arr = np.asarray(stationary, dtype=np.float64)
    if stationary_arr.ndim != 1:
        raise ValueError("stationary must be a one-dimensional array")
    if len(policy_ids) != stationary_arr.shape[0]:
        raise ValueError("policy_ids length must match stationary length")

    ranked_rows = sorted(
        zip(policy_ids, stationary_arr.tolist(), strict=True),
        key=lambda item: (-item[1], item[0]),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "policy_id", "stationary_probability"])
        for rank, (policy_id, prob) in enumerate(ranked_rows, start=1):
            writer.writerow([rank, policy_id, f"{float(prob):.12g}"])


def write_alpharank_artifacts(
    stationary_mean_csv: Path,
    stationary: np.ndarray,
    policy_ids: Sequence[str],
) -> None:
    """Write AlphaRank artifacts to files."""
    write_stationary_mean_csv(stationary_mean_csv, policy_ids, stationary)


def _build_transition_matrix_local(
    p_mean: np.ndarray,
    *,
    m: int,
    alpha: float,
    use_inf_alpha: bool,
    tie_tolerance: float,
) -> np.ndarray:
    return _build_transition_matrix(
        p_mean,
        m=m,
        alpha=alpha,
        use_inf_alpha=use_inf_alpha,
        tie_tolerance=tie_tolerance,
        delta_profile_fn=_local_selection_profile,
    )


def _build_transition_matrix_global(
    p_mean: np.ndarray,
    *,
    m: int,
    alpha: float,
    use_inf_alpha: bool,
    tie_tolerance: float,
) -> np.ndarray:
    return _build_transition_matrix(
        p_mean,
        m=m,
        alpha=alpha,
        use_inf_alpha=use_inf_alpha,
        tie_tolerance=tie_tolerance,
        delta_profile_fn=_global_selection_profile,
    )


def _build_transition_matrix(
    p_mean: np.ndarray,
    *,
    m: int,
    alpha: float,
    use_inf_alpha: bool,
    tie_tolerance: float,
    delta_profile_fn,
) -> np.ndarray:
    num_policies = p_mean.shape[0]
    transition_matrix = np.zeros((num_policies, num_policies), dtype=np.float64)
    mutation_mass = 1.0 / float(num_policies - 1)

    for resident_index in range(num_policies):
        row_mass = 0.0
        for mutant_index in range(num_policies):
            if mutant_index == resident_index:
                continue
            deltas = delta_profile_fn(
                p_mean,
                resident_index=resident_index,
                mutant_index=mutant_index,
                m=m,
            )
            fixation = _pairwise_fixation_probability(
                deltas=deltas,
                m=m,
                alpha=alpha,
                use_inf_alpha=use_inf_alpha,
                tie_tolerance=tie_tolerance,
            )
            transition_probability = mutation_mass * fixation
            transition_matrix[resident_index, mutant_index] = transition_probability
            row_mass += transition_probability

        self_loop = 1.0 - row_mass
        if self_loop < 0.0 and self_loop > -_TRANSITION_ATOL:
            self_loop = 0.0
        if self_loop < 0.0:
            raise RuntimeError("AlphaRank transition row mass exceeded 1")
        transition_matrix[resident_index, resident_index] = self_loop

    row_sums = np.sum(transition_matrix, axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, atol=_TRANSITION_ATOL, rtol=0.0):
        raise RuntimeError("AlphaRank transition matrix must be row-stochastic")
    return transition_matrix


def _local_selection_delta(p_mean: np.ndarray, *, resident_index: int, mutant_index: int) -> float:
    return float(p_mean[mutant_index, resident_index] - p_mean[resident_index, mutant_index])


def _local_selection_profile(
    p_mean: np.ndarray,
    *,
    resident_index: int,
    mutant_index: int,
    m: int,
) -> np.ndarray:
    if m <= 1:
        return np.zeros((0,), dtype=np.float64)
    delta = _local_selection_delta(p_mean, resident_index=resident_index, mutant_index=mutant_index)
    return np.full((m - 1,), delta, dtype=np.float64)


def _global_selection_profile(
    p_mean: np.ndarray,
    *,
    resident_index: int,
    mutant_index: int,
    m: int,
) -> np.ndarray:
    if m <= 1:
        return np.zeros((0,), dtype=np.float64)
    if m == 2:
        delta = _local_selection_delta(p_mean, resident_index=resident_index, mutant_index=mutant_index)
        return np.asarray([delta], dtype=np.float64)

    deltas = np.zeros((m - 1,), dtype=np.float64)
    denom = float(m - 1)
    for mutant_count in range(1, m):
        resident_count = m - mutant_count
        mutant_fitness = (
            ((mutant_count - 1) * p_mean[mutant_index, mutant_index])
            + (resident_count * p_mean[mutant_index, resident_index])
        ) / denom
        resident_fitness = (
            (mutant_count * p_mean[resident_index, mutant_index])
            + ((resident_count - 1) * p_mean[resident_index, resident_index])
        ) / denom
        deltas[mutant_count - 1] = float(mutant_fitness - resident_fitness)
    return deltas


def _pairwise_fixation_probability(
    *,
    deltas: np.ndarray,
    m: int,
    alpha: float,
    use_inf_alpha: bool,
    tie_tolerance: float,
) -> float:
    if m == 1:
        return 1.0

    delta_profile = np.asarray(deltas, dtype=np.float64)
    if delta_profile.shape != (m - 1,):
        raise ValueError(f"deltas must have shape ({m - 1},), got {delta_profile.shape}")
    cumulative_deltas = np.cumsum(delta_profile, dtype=np.float64)

    if use_inf_alpha:
        if np.any(cumulative_deltas < -tie_tolerance):
            return 0.0
        zero_like = int(np.count_nonzero(np.abs(cumulative_deltas) <= tie_tolerance))
        if zero_like:
            return 1.0 / float(1 + zero_like)
        return 1.0

    if np.all(np.abs(cumulative_deltas) <= tie_tolerance):
        return 1.0 / float(m)

    exponents = -(float(alpha) * np.concatenate((np.asarray([0.0], dtype=np.float64), cumulative_deltas)))
    max_exponent = float(np.max(exponents))
    shifted_sum = float(np.sum(np.exp(exponents - max_exponent), dtype=np.float64))
    log_denom = max_exponent + float(np.log(shifted_sum))
    return float(np.exp(-log_denom))


def _solve_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    num_policies = transition_matrix.shape[0]
    system: np.ndarray = transition_matrix.T - np.eye(num_policies, dtype=np.float64)
    system[-1, :] = 1.0
    rhs: np.ndarray = np.zeros(num_policies, dtype=np.float64)
    rhs[-1] = 1.0

    try:
        stationary = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        stationary = _power_iteration_stationary(transition_matrix)
    else:
        if not np.isfinite(stationary).all() or np.any(stationary < -_TRANSITION_ATOL):
            stationary = _power_iteration_stationary(transition_matrix)

    clipped = np.clip(stationary, a_min=0.0, a_max=None)
    return normalize_stationary(clipped)


def _power_iteration_stationary(transition_matrix: np.ndarray) -> np.ndarray:
    num_policies = transition_matrix.shape[0]
    stationary = np.full(num_policies, 1.0 / float(num_policies), dtype=np.float64)
    for _ in range(_STATIONARY_MAX_ITERS):
        next_stationary = stationary @ transition_matrix
        if float(np.linalg.norm(next_stationary - stationary, ord=1)) <= _STATIONARY_TOL:
            return next_stationary
        stationary = next_stationary
    raise RuntimeError("AlphaRank stationary distribution failed to converge")
