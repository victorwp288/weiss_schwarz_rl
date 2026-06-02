"""Parser construction for the paper-readiness check CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from weiss_rl.eval.paper_readiness import (
    DEFAULT_BASELINE_POLICY_ID,
    DEFAULT_BASELINE_POSTERIOR_MIN,
    DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
    DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
    DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
    DEFAULT_TRUNCATION_MAX_RATE,
)


def closed_interval(*, lower: float, upper: float, label: str) -> Callable[[str], float]:
    def _parse(value: str) -> float:
        parsed = float(value)
        if parsed < lower or parsed > upper:
            raise argparse.ArgumentTypeError(f"{label} must be in [{lower}, {upper}]")
        return parsed

    return _parse


def default_readiness_json(*, run_dir: Path | None, final_eval_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir / "paper_readiness_summary.json"
    assert final_eval_dir is not None
    return final_eval_dir / "paper_readiness_summary.json"


def build_paper_readiness_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-readiness audit over run directories and final_eval artifacts")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a run directory to audit for paper-grade submission artifacts",
    )
    target_group.add_argument(
        "--final-eval-dir",
        type=Path,
        default=None,
        help=(
            "Compatibility mode, path to a final_eval artifact directory "
            "containing summary.json and matchup diagnostics"
        ),
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        default=None,
        help=(
            "Output path for the readiness summary JSON "
            "(default: <run-dir>/paper_readiness_summary.json or <final-eval-dir>/paper_readiness_summary.json)"
        ),
    )
    parser.add_argument(
        "--focal-policy-id",
        type=str,
        default="",
        help=(
            "Policy to check against the B0 baseline "
            "(default: auto-resolve only when exactly one eligible non-baseline policy exists, "
            "or metadata names the focal policy explicitly)"
        ),
    )
    parser.add_argument(
        "--baseline-policy-id",
        type=str,
        default=DEFAULT_BASELINE_POLICY_ID,
        help="Baseline policy ID used for the win-rate guardrail",
    )
    parser.add_argument(
        "--max-truncation-rate",
        type=closed_interval(lower=0.0, upper=1.0, label="--max-truncation-rate"),
        default=DEFAULT_TRUNCATION_MAX_RATE,
        help="Maximum allowed aggregate truncation rate across canonical unordered final_eval matchups",
    )
    parser.add_argument(
        "--seat-bias-max-abs-delta",
        type=closed_interval(lower=0.0, upper=0.5, label="--seat-bias-max-abs-delta"),
        default=DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
        help="Seat-bias alarm margin around 0.5 for the global decisive seat0 win rate",
    )
    parser.add_argument(
        "--seat-bias-posterior-min",
        type=closed_interval(lower=0.0, upper=1.0, label="--seat-bias-posterior-min"),
        default=DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
        help="Posterior probability threshold for triggering the seat-bias alarm",
    )
    parser.add_argument(
        "--baseline-win-rate-threshold",
        type=closed_interval(lower=0.0, upper=1.0, label="--baseline-win-rate-threshold"),
        default=DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
        help="Minimum posterior win-rate threshold for the focal policy versus the baseline",
    )
    parser.add_argument(
        "--baseline-posterior-min",
        type=closed_interval(lower=0.0, upper=1.0, label="--baseline-posterior-min"),
        default=DEFAULT_BASELINE_POSTERIOR_MIN,
        help="Required posterior probability of exceeding the baseline win-rate threshold",
    )
    return parser
