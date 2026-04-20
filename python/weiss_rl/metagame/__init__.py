"""Metagame analysis utilities."""

from .alpharank import (
    AlphaRankResult,
    compute_alpharank_stationary,
    compute_stationary_distribution,
    normalize_stationary,
    write_alpharank_artifacts,
    write_stationary_mean_csv,
)
from .nash import (
    NashSolveResult,
    NashSolverReport,
    solve_nash_mixture,
    solve_zero_sum_mixture,
    uniform_mixture,
    write_mixture_mean_csv,
    write_nash_artifacts,
    write_solver_report_json,
)
from .payoff import (
    build_p_mean_and_counts,
    to_antisymmetric,
    write_p_mean_csv,
    write_payoff_artifacts,
    write_payoff_counts_json,
)
from .sensitivity import build_sensitivity_report
from .uncertainty import (
    PayoffUncertaintySummary,
    bayesian_bootstrap_summary,
    dirichlet_wldt_posterior_samples,
    dirichlet_wldt_posterior_summary,
    optional_secondary_uncertainty_summary,
    paired_seed_uncertainty_summary,
    posterior_samples,
    write_posterior_samples,
    write_uncertainty_artifacts,
    write_uncertainty_summary_json,
)

__all__ = [
    "AlphaRankResult",
    "NashSolveResult",
    "NashSolverReport",
    "PayoffUncertaintySummary",
    "bayesian_bootstrap_summary",
    "build_p_mean_and_counts",
    "build_sensitivity_report",
    "compute_alpharank_stationary",
    "compute_stationary_distribution",
    "normalize_stationary",
    "paired_seed_uncertainty_summary",
    "optional_secondary_uncertainty_summary",
    "dirichlet_wldt_posterior_summary",
    "dirichlet_wldt_posterior_samples",
    "posterior_samples",
    "solve_nash_mixture",
    "solve_zero_sum_mixture",
    "to_antisymmetric",
    "uniform_mixture",
    "write_alpharank_artifacts",
    "write_mixture_mean_csv",
    "write_nash_artifacts",
    "write_p_mean_csv",
    "write_payoff_artifacts",
    "write_payoff_counts_json",
    "write_posterior_samples",
    "write_solver_report_json",
    "write_stationary_mean_csv",
    "write_uncertainty_artifacts",
    "write_uncertainty_summary_json",
]
