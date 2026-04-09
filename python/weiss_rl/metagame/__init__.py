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
from .uncertainty import (
    PayoffUncertaintySummary,
    bayesian_bootstrap_summary,
    paired_seed_uncertainty_summary,
    posterior_samples,
    write_posterior_samples,
    write_uncertainty_artifacts,
    write_uncertainty_summary_json,
)

__all__ = [
    "AlphaRankResult",
    "NashSolveResult",
    "PayoffUncertaintySummary",
    "bayesian_bootstrap_summary",
    "build_p_mean_and_counts",
    "build_sensitivity_report",
    "compute_stationary_distribution",
    "normalize_stationary",
    "paired_seed_uncertainty_summary",
    "posterior_samples",
    "solve_zero_sum_mixture",
    "to_antisymmetric",
    "uniform_mixture",
    "write_p_mean_csv",
    "write_payoff_artifacts",
    "write_payoff_counts_json",
    "write_posterior_samples",
    "write_uncertainty_artifacts",
    "write_uncertainty_summary_json",
]
