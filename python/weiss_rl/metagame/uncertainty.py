"""Uncertainty estimation helpers for metagame payoff posterior analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from weiss_rl.eval import EvalGameRecord
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.uncertainty import (
    EvalUncertaintySummary,
    bayesian_bootstrap_summary as eval_bayesian_bootstrap_summary,
    paired_seed_uncertainty_summary as eval_paired_seed_uncertainty_summary,
    posterior_samples as eval_posterior_samples,
)

_DEFAULT_CI_LEVEL = 0.95
_DEFAULT_SAMPLE_COUNT = 1000

__all__ = [
    "PayoffUncertaintySummary",
    "bayesian_bootstrap_summary",
    "paired_seed_uncertainty_summary",
    "posterior_samples",
    "write_posterior_samples",
    "write_uncertainty_summary_json",
    "write_uncertainty_artifacts",
]


@dataclass(frozen=True, slots=True)
class PayoffUncertaintySummary:
    mean: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    prob_gt_half: float
    prob_lt_half: float
    paired_seed_count: int
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bayesian_bootstrap_summary(
    scores: Sequence[float],
    *,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> PayoffUncertaintySummary:
    return _from_eval_summary(
        eval_bayesian_bootstrap_summary(
            scores,
            sample_count=sample_count,
            ci_level=ci_level,
            seed=seed,
        )
    )


def paired_seed_uncertainty_summary(
    records: Sequence[EvalGameRecord],
    *,
    scheme: PayoffFoldScheme,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> PayoffUncertaintySummary:
    return _from_eval_summary(
        eval_paired_seed_uncertainty_summary(
            records,
            scheme=scheme,
            sample_count=sample_count,
            ci_level=ci_level,
            seed=seed,
        )
    )


def posterior_samples(
    scores: Sequence[float] | np.ndarray, *, sample_count: int = _DEFAULT_SAMPLE_COUNT, seed: int | None = None
) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    return eval_posterior_samples(score_array.tolist(), sample_count=sample_count, seed=seed)


def write_posterior_samples(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, posterior_samples=np.asarray(samples, dtype=np.float64))


def write_uncertainty_summary_json(path: Path, summary: PayoffUncertaintySummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_uncertainty_artifacts(
    samples_path: Path,
    summary_path: Path,
    summary: PayoffUncertaintySummary,
    samples: np.ndarray,
) -> None:
    write_posterior_samples(samples_path, samples)
    write_uncertainty_summary_json(summary_path, summary)


def _from_eval_summary(summary: EvalUncertaintySummary) -> PayoffUncertaintySummary:
    return PayoffUncertaintySummary(
        mean=summary.mean,
        ci_low=summary.ci_low,
        ci_high=summary.ci_high,
        ci_half_width=summary.ci_half_width,
        prob_gt_half=summary.prob_gt_half,
        prob_lt_half=summary.prob_lt_half,
        paired_seed_count=summary.paired_seed_count,
        sample_count=summary.sample_count,
    )
