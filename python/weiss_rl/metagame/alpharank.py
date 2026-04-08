"""Single-population AlphaRank helpers."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "AlphaRankResult",
    "compute_alpharank_stationary",
    "compute_stationary_distribution",
    "normalize_stationary",
    "write_alpharank_artifacts",
    "write_stationary_mean_csv",
]


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
    """Compute the AlphaRank stationary distribution for policy ranking."""

    p_mean_arr = np.asarray(p_mean, dtype=np.float64)
    if p_mean_arr.ndim != 2 or p_mean_arr.shape[0] != p_mean_arr.shape[1]:
        raise ValueError("p_mean must be a square matrix")
    if p_mean_arr.shape[0] == 0:
        raise ValueError("p_mean must contain at least one policy")
    if policy_ids is not None and len(policy_ids) != p_mean_arr.shape[0]:
        raise ValueError("policy_ids length must match p_mean dimensions")
    if not np.isfinite(p_mean_arr).all():
        raise ValueError("p_mean must contain only finite values")
    if m < 1:
        raise ValueError("m must be >= 1")
    if alpha < 1:
        raise ValueError("alpha must be >= 1")
    if inf_alpha_eps < 0.0:
        raise ValueError("inf_alpha_eps must be >= 0")
    if p_mean_arr.shape[0] == 1:
        return np.ones(1, dtype=np.float64)

    return compute_stationary_distribution(
        p_mean_arr,
        m=m,
        alpha=alpha,
        local_selection=local_selection,
        use_inf_alpha=use_inf_alpha,
        inf_alpha_eps=inf_alpha_eps,
    ).stationary


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
    write_stationary_mean_csv(stationary_mean_csv, policy_ids, stationary)


def compute_stationary_distribution(
    payoff: np.ndarray,
    *,
    m: int,
    alpha: int,
    local_selection: bool,
    use_inf_alpha: bool,
    inf_alpha_eps: float,
) -> AlphaRankResult:
    matrix = _validate_payoff(payoff)
    policy_count = matrix.shape[0]
    if policy_count == 1:
        stationary = np.asarray([1.0], dtype=np.float64)
        return AlphaRankResult(stationary=stationary, transition_matrix=np.asarray([[1.0]], dtype=np.float64))

    transition = np.zeros((policy_count, policy_count), dtype=np.float64)
    for resident in range(policy_count):
        off_diagonal_total = 0.0
        for mutant in range(policy_count):
            if resident == mutant:
                continue
            fixation = _fixation_probability(
                matrix,
                resident=resident,
                mutant=mutant,
                m=m,
                alpha=float(alpha),
                local_selection=local_selection,
                use_inf_alpha=use_inf_alpha,
                inf_alpha_eps=inf_alpha_eps,
            )
            weight = fixation / float(policy_count - 1)
            transition[resident, mutant] = weight
            off_diagonal_total += weight
        transition[resident, resident] = max(0.0, 1.0 - off_diagonal_total)
    stationary = _power_iteration_stationary(transition)
    return AlphaRankResult(stationary=stationary, transition_matrix=transition)


def _validate_payoff(payoff: np.ndarray) -> np.ndarray:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("payoff must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("payoff must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff must be finite")
    return matrix


def _fixation_probability(
    payoff: np.ndarray,
    *,
    resident: int,
    mutant: int,
    m: int,
    alpha: float,
    local_selection: bool,
    use_inf_alpha: bool,
    inf_alpha_eps: float,
) -> float:
    if m <= 1:
        raise ValueError("AlphaRank population size m must be > 1")
    if alpha <= 0.0:
        raise ValueError("AlphaRank alpha must be > 0")

    differences = np.asarray(
        [
            _fitness_difference(
                payoff,
                resident=resident,
                mutant=mutant,
                mutant_count=mutant_count,
                population_size=m,
                local_selection=local_selection,
            )
            for mutant_count in range(1, m)
        ],
        dtype=np.float64,
    )
    if use_inf_alpha:
        decisive = float(np.sum(differences))
        if decisive > inf_alpha_eps:
            return 1.0
        if decisive < -inf_alpha_eps:
            return 0.0
        return 1.0 / float(m)

    cumulative = np.cumsum(differences)
    log_terms = np.concatenate((np.asarray([0.0], dtype=np.float64), -alpha * cumulative))
    return float(np.exp(-logsumexp(log_terms)))


def _fitness_difference(
    payoff: np.ndarray,
    *,
    resident: int,
    mutant: int,
    mutant_count: int,
    population_size: int,
    local_selection: bool,
) -> float:
    if local_selection:
        resident_payoff = _population_payoff(
            payoff,
            strategy=resident,
            other_strategy=mutant,
            strategy_count=population_size - mutant_count,
            population_size=population_size,
        )
        mutant_payoff = _population_payoff(
            payoff,
            strategy=mutant,
            other_strategy=resident,
            strategy_count=mutant_count,
            population_size=population_size,
        )
        return mutant_payoff - resident_payoff
    return float(payoff[mutant, resident] - payoff[resident, mutant])


def _population_payoff(
    payoff: np.ndarray,
    *,
    strategy: int,
    other_strategy: int,
    strategy_count: int,
    population_size: int,
) -> float:
    own_weight = float(max(strategy_count - 1, 0)) / float(population_size - 1)
    other_weight = float(population_size - strategy_count) / float(population_size - 1)
    return float((own_weight * payoff[strategy, strategy]) + (other_weight * payoff[strategy, other_strategy]))


def _power_iteration_stationary(transition: np.ndarray, *, max_iter: int = 10000, tol: float = 1.0e-12) -> np.ndarray:
    policy_count = transition.shape[0]
    state = np.full((policy_count,), 1.0 / float(policy_count), dtype=np.float64)
    for _ in range(max_iter):
        updated = state @ transition
        if np.max(np.abs(updated - state)) <= tol:
            return normalize_stationary(updated)
        state = updated
    raise RuntimeError("AlphaRank power iteration did not converge")
