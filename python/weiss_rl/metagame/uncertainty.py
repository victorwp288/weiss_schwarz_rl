"""Uncertainty estimation helpers for metagame payoff posterior analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from weiss_rl.eval import EvalGameRecord
from weiss_rl.eval.payoff_folding import (
    PayoffFoldScheme,
    _normalize_scheme,
    paired_seed_scores,
    validated_paired_seed_groups,
)

_DEFAULT_CI_LEVEL = 0.95
_DEFAULT_SAMPLE_COUNT = 1000
_DECISIVE_THRESHOLD = 0.5
_OPTIONAL_SECONDARY_UNCERTAINTY_METHOD_DIRICHLET_WLDT_JEFFERYS_V1 = "dirichlet_wldt_jeffreys_v1"

__all__ = [
    "PayoffUncertaintySummary",
    "bayesian_bootstrap_summary",
    "paired_seed_uncertainty_summary",
    "optional_secondary_uncertainty_summary",
    "dirichlet_wldt_posterior_summary",
    "dirichlet_wldt_posterior_samples",
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
    posterior = posterior_samples(score_array.tolist(), sample_count=sample_count, seed=seed)
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
    score_array = _coerce_scores(scores.tolist() if isinstance(scores, np.ndarray) else scores)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    rng = np.random.default_rng(seed)
    weights = rng.exponential(scale=1.0, size=(sample_count, score_array.size))
    weights /= np.sum(weights, axis=1, keepdims=True)
    baseline = float(score_array[0])
    return baseline + (weights @ (score_array - baseline))


def optional_secondary_uncertainty_summary(
    records: Sequence[EvalGameRecord],
    *,
    scheme: PayoffFoldScheme,
    method: str = _OPTIONAL_SECONDARY_UNCERTAINTY_METHOD_DIRICHLET_WLDT_JEFFERYS_V1,
    dirichlet_alpha_wldt: float = 0.5,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> PayoffUncertaintySummary:
    if method == _OPTIONAL_SECONDARY_UNCERTAINTY_METHOD_DIRICHLET_WLDT_JEFFERYS_V1:
        return dirichlet_wldt_posterior_summary(
            records,
            scheme=scheme,
            alpha=dirichlet_alpha_wldt,
            sample_count=sample_count,
            ci_level=ci_level,
            seed=seed,
        )
    raise ValueError(f"unknown optional secondary uncertainty method: {method!r}")


def dirichlet_wldt_posterior_summary(
    records: Sequence[EvalGameRecord],
    *,
    scheme: PayoffFoldScheme,
    alpha: float = 0.5,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> PayoffUncertaintySummary:
    samples = dirichlet_wldt_posterior_samples(
        records, scheme=scheme, alpha=alpha, sample_count=sample_count, seed=seed
    )
    ci_low, ci_high = _credible_interval(samples, ci_level=ci_level)
    return PayoffUncertaintySummary(
        mean=float(np.mean(samples)),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_half_width=(ci_high - ci_low) / 2.0,
        prob_gt_half=float(np.mean(samples > _DECISIVE_THRESHOLD)),
        prob_lt_half=float(np.mean(samples < _DECISIVE_THRESHOLD)),
        paired_seed_count=len(validated_paired_seed_groups(records)),
        sample_count=sample_count,
    )


def dirichlet_wldt_posterior_samples(
    records: Sequence[EvalGameRecord],
    *,
    scheme: str,
    alpha: float = 0.5,
    sample_count: int = _DEFAULT_SAMPLE_COUNT,
    seed: int | None = None,
) -> np.ndarray:
    if not records:
        raise ValueError("dirichlet_wldt_posterior_samples requires at least one record")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    normalized_scheme = _normalize_scheme(scheme)
    counts = _count_wldt_outcomes(records)
    rng = np.random.default_rng(seed)
    theta = rng.dirichlet(counts + alpha, size=sample_count)
    if normalized_scheme in ("S0", "S1"):
        return theta[:, 0] + 0.5 * (theta[:, 2] + theta[:, 3])

    nontrunc = theta[:, :3]
    nontrunc_mass = np.sum(nontrunc, axis=1)
    scores = np.empty(sample_count, dtype=np.float64)
    for sample_index in range(sample_count):
        if nontrunc_mass[sample_index] > 0.0:
            scores[sample_index] = nontrunc[sample_index, 0] / nontrunc_mass[sample_index]
            scores[sample_index] += 0.5 * (nontrunc[sample_index, 2] / nontrunc_mass[sample_index])
        else:
            scores[sample_index] = 0.5
    return scores


def _count_wldt_outcomes(records: Sequence[EvalGameRecord]) -> np.ndarray:
    counts = np.zeros((4,), dtype=np.float64)
    for pair_records in validated_paired_seed_groups(records):
        for record in pair_records:
            outcome = record.outcome.strip().upper()
            if outcome == "W":
                counts[0] += 1.0
            elif outcome == "L":
                counts[1] += 1.0
            elif outcome == "D":
                counts[2] += 1.0
            elif outcome == "T":
                counts[3] += 1.0
            else:
                raise ValueError(f"unknown outcome token: {record.outcome!r}")
    return counts


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


def _coerce_scores(scores: Sequence[float]) -> np.ndarray:
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
