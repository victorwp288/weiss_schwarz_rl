"""Bayesian bootstrap uncertainty over paired-seed evaluation scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores

_DEFAULT_CI_LEVEL = 0.95
_DEFAULT_SAMPLE_COUNT = 1000
_DECISIVE_THRESHOLD = 0.5

__all__ = [
    "EvalUncertaintySummary",
    "bayesian_bootstrap_posterior_samples",
    "bayesian_bootstrap_summary",
    "paired_seed_uncertainty_summary",
    "posterior_samples",
]


@dataclass(frozen=True, slots=True)
class EvalUncertaintySummary:
    mean: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    prob_gt_half: float
    prob_lt_half: float
    paired_seed_count: int
    sample_count: int


def bayesian_bootstrap_posterior_samples(
    scores: Sequence[float],
    *,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    seed: int | None = None,
) -> tuple[float, ...]:
    score_array = _coerce_scores(scores)
    posterior_samples = _posterior_samples_from_array(score_array, sample_count=sample_count, seed=seed)
    return tuple(float(sample) for sample in posterior_samples)


def bayesian_bootstrap_summary(
    scores: Sequence[float],
    *,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> EvalUncertaintySummary:
    score_array = _coerce_scores(scores)
    samples = _posterior_samples_from_array(score_array, sample_count=sample_count, seed=seed)
    ci_low, ci_high = _credible_interval(samples, ci_level=ci_level)
    mean = _mean(score_array)
    return EvalUncertaintySummary(
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_half_width=(ci_high - ci_low) / 2.0,
        prob_gt_half=float(np.mean(samples > _DECISIVE_THRESHOLD)),
        prob_lt_half=float(np.mean(samples < _DECISIVE_THRESHOLD)),
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
) -> EvalUncertaintySummary:
    normalized_scheme = scheme.strip().upper()
    pair_scores = paired_seed_scores(records, scheme=scheme)
    if pair_scores:
        return bayesian_bootstrap_summary(
            pair_scores,
            sample_count=sample_count,
            ci_level=ci_level,
            seed=seed,
        )
    raise ValueError(f"{normalized_scheme} excluded all paired seeds")


def _coerce_scores(scores: Sequence[float]) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("bayesian_bootstrap_summary requires at least one score")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    return score_array


def posterior_samples(
    scores: Sequence[float], *, sample_count: int = _DEFAULT_SAMPLE_COUNT, seed: int | None = None
) -> np.ndarray:
    score_array = _coerce_scores(scores)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    return _posterior_samples_from_array(score_array, sample_count=sample_count, seed=seed)


def _posterior_samples_from_array(scores: np.ndarray, *, sample_count: int, seed: int | None) -> np.ndarray:
    baseline = float(scores[0])
    if scores.size == 1 or np.all(scores == baseline):
        return np.full((sample_count,), baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    weights = rng.exponential(scale=1.0, size=(sample_count, scores.size))
    weights /= np.sum(weights, axis=1, keepdims=True)
    centered = np.asarray(scores - baseline, dtype=np.float64)
    return baseline + np.sum(weights * centered[np.newaxis, :], axis=1, dtype=np.float64)


def _credible_interval(samples: np.ndarray, *, ci_level: float) -> tuple[float, float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    alpha = 1.0 - ci_level
    ci_low = float(np.quantile(samples, alpha / 2.0))
    ci_high = float(np.quantile(samples, 1.0 - (alpha / 2.0)))
    return ci_low, ci_high


def _mean(scores: np.ndarray) -> float:
    return float(np.sum(scores) / scores.size)
