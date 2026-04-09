"""Metagame analysis utilities."""

from .alpharank import AlphaRankResult, compute_stationary_distribution, normalize_stationary
from .nash import NashSolveResult, solve_zero_sum_mixture, uniform_mixture
from .payoff import (
    build_p_mean_and_counts,
    write_p_mean_csv,
    write_payoff_artifacts,
    write_payoff_counts_json,
    to_antisymmetric,
)
from .sensitivity import build_sensitivity_report

__all__ = [
    "AlphaRankResult",
    "NashSolveResult",
    "build_p_mean_and_counts",
    "build_sensitivity_report",
    "compute_stationary_distribution",
    "normalize_stationary",
    "solve_zero_sum_mixture",
    "to_antisymmetric",
    "uniform_mixture",
    "write_p_mean_csv",
    "write_payoff_artifacts",
    "write_payoff_counts_json",
]
