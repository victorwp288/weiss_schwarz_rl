"""Uncertainty estimation helpers for metagame payoff posterior analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from weiss_rl.eval import EvalGameRecord
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores

_DEFAULT_CI_LEVEL = 0.95
_DEFAULT_SAMPLE_COUNT = 1000
_DECISIVE_THRESHOLD = 0.5

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
    score_array = _coerce_scores(scores)
    posterior = posterior_samples(score_array, sample_count=sample_count, seed=seed)
    ci_low, ci_high = _credible_interval(posterior, ci_level=ci_level)
    mean = float(np.mean(score_array))
    return PayoffUncertaintySummary(
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_half_width=(ci_high - ci_low) / 2.0,
        prob_gt_half=float(np.mean(posterior > _DECISIVE_THRESHOLD)),
        prob_lt_half=float(np.mean(posterior < _DECISIVE_THRESHOLD)),
        paired_seed_count=int(score_array.size),
        sample_count=sample_count,
    )


def paired_seed_uncertainty_summary(
    records: Sequence[EvalGameRecord],
    *,
    scheme: PayoffFoldScheme,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> PayoffUncertaintySummary:
    if not records:
        raise ValueError("paired_seed_uncertainty_summary requires at least one record")
    pair_scores = paired_seed_scores(records, scheme=scheme)
    if not pair_scores:
        raise ValueError(f"{scheme} excluded all paired seeds")
    return bayesian_bootstrap_summary(
        pair_scores,
        sample_count=sample_count,
        ci_level=ci_level,
        seed=seed,
    )


def posterior_samples(
    scores: Sequence[float] | np.ndarray, *, sample_count: int = _DEFAULT_SAMPLE_COUNT, seed: int | None = None
) -> np.ndarray:
    score_array = _coerce_scores(scores)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    rng = np.random.default_rng(seed)
    weights = rng.exponential(scale=1.0, size=(sample_count, score_array.size))
    weights /= np.sum(weights, axis=1, keepdims=True)
    baseline = float(score_array[0])
    return baseline + (weights @ (score_array - baseline))


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


def _coerce_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    return score_array


def _credible_interval(samples: np.ndarray, *, ci_level: float) -> tuple[float, float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")
    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(samples, alpha / 2.0))
    ci_high = float(np.quantile(samples, 1.0 - (alpha / 2.0)))
    return ci_low, ci_high
