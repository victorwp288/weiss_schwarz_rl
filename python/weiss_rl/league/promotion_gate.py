"""Promotion gate runner for AnchorSet_v1."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.artifacts.reproducibility import hash_seed_file, parse_seed_file
from weiss_rl.config import LeagueConfig, StackConfig
from weiss_rl.eval import bayesian_bootstrap_summary, paired_seed_scores, run_seat_swapped_matchup
from weiss_rl.eval.harness import EvalGameRunner, EvalRunResult, MatchupSummary

_PROMOTION_MEAN_THRESHOLD = 0.55
_PROMOTION_LOSS_THRESHOLD = 0.45
_PROMOTION_POSTERIOR_MIN = 0.95
_DEFAULT_BOOTSTRAP_SAMPLE_COUNT = 1000
_ANCHOR_EPISODES_DIRNAME = "promotion_gate_episodes"

__all__ = [
    "PromotionGateAnchor",
    "PromotionGateAnchorResult",
    "PromotionGatePosterior",
    "PromotionGateRate",
    "PromotionGateResult",
    "resolve_promotion_gate_anchors",
    "resolve_promotion_gate_seed_file",
    "run_promotion_gate",
]


@dataclass(frozen=True, slots=True)
class PromotionGateAnchor:
    name: str
    policy_id: str


@dataclass(frozen=True, slots=True)
class PromotionGateRate:
    numerator: int
    denominator: int
    rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionGatePosterior:
    mean: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    prob_gt_half: float
    prob_lt_half: float
    prob_gt_target: float
    prob_lt_guardrail: float
    paired_seed_count: int
    sample_count: int
    gt_target: float = _PROMOTION_MEAN_THRESHOLD
    lt_guardrail: float = _PROMOTION_LOSS_THRESHOLD

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[float],
        *,
        sample_count: int,
        seed: int | None,
    ) -> PromotionGatePosterior:
        summary = bayesian_bootstrap_summary(scores, sample_count=sample_count, seed=seed)
        posterior_samples = _posterior_means(scores, sample_count=sample_count, seed=seed)
        return cls(
            mean=summary.mean,
            ci_low=summary.ci_low,
            ci_high=summary.ci_high,
            ci_half_width=summary.ci_half_width,
            prob_gt_half=summary.prob_gt_half,
            prob_lt_half=summary.prob_lt_half,
            prob_gt_target=float(np.mean(posterior_samples > _PROMOTION_MEAN_THRESHOLD)),
            prob_lt_guardrail=float(np.mean(posterior_samples < _PROMOTION_LOSS_THRESHOLD)),
            paired_seed_count=summary.paired_seed_count,
            sample_count=summary.sample_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionGateAnchorResult:
    anchor_name: str
    opponent_policy_id: str
    episodes_path: str
    matchup_summary: MatchupSummary
    truncation: PromotionGateRate
    posterior: PromotionGatePosterior

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_name": self.anchor_name,
            "episodes_path": self.episodes_path,
            "matchup_summary": asdict(self.matchup_summary),
            "opponent_policy_id": self.opponent_policy_id,
            "posterior": self.posterior.to_dict(),
            "truncation": self.truncation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    focal_policy_id: str
    ordered_opponents: tuple[str, ...]
    record_path: str
    seed_file_path: str
    seed_file_sha256: str
    paired_seed_count: int
    weighting: str
    seat_swap: bool
    folding: str
    anchors: tuple[PromotionGateAnchorResult, ...]
    overall_posterior: PromotionGatePosterior
    truncation: PromotionGateRate
    passed: bool
    reasons: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "decision": {
                "passed": self.passed,
                "reasons": list(self.reasons),
            },
            "focal_policy_id": self.focal_policy_id,
            "folding": self.folding,
            "ordered_opponents": list(self.ordered_opponents),
            "overall_posterior": self.overall_posterior.to_dict(),
            "paired_seed_count": self.paired_seed_count,
            "record_path": self.record_path,
            "seat_swap": self.seat_swap,
            "seed_file": {
                "path": self.seed_file_path,
                "sha256": self.seed_file_sha256,
            },
            "truncation": self.truncation.to_dict(),
            "weighting": self.weighting,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_promotion_gate(
    *,
    stack: StackConfig,
    run_dir: Path,
    focal_policy_id: str,
    anchor_policy_ids: Mapping[str, str],
    runner: EvalGameRunner,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    sample_count: int = _DEFAULT_BOOTSTRAP_SAMPLE_COUNT,
    bootstrap_seed: int | None = None,
) -> PromotionGateResult:
    league = _require_league_config(stack)
    _validate_gate_contract(stack)

    record_path = run_dir / league.promotion_gate.record_file
    episodes_dir = record_path.parent / _ANCHOR_EPISODES_DIRNAME
    seed_file = resolve_promotion_gate_seed_file(stack)
    paired_seeds = parse_seed_file(seed_file)
    _validate_seed_count(paired_seeds, expected=league.promotion_gate_paired_seeds, seed_file=seed_file)
    anchors = resolve_promotion_gate_anchors(stack, anchor_policy_ids)

    anchor_results: list[PromotionGateAnchorResult] = []
    all_pair_scores: list[float] = []
    total_truncated_games = 0
    total_games = 0

    for anchor_index, anchor in enumerate(anchors):
        matchup = run_seat_swapped_matchup(
            focal_policy_id=focal_policy_id,
            opponent_policy_id=anchor.policy_id,
            paired_seeds=paired_seeds,
            runner=runner,
            episodes_path=episodes_dir / f"{anchor_index:02d}_{_slug(anchor.name)}.jsonl",
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
        )
        anchor_result = _build_anchor_result(
            anchor=anchor,
            matchup=matchup,
            run_dir=run_dir,
            sample_count=sample_count,
            bootstrap_seed=bootstrap_seed,
        )
        anchor_results.append(anchor_result)
        all_pair_scores.extend(paired_seed_scores(matchup.records, scheme="S0"))
        total_truncated_games += anchor_result.truncation.numerator
        total_games += anchor_result.truncation.denominator

    overall_posterior = PromotionGatePosterior.from_scores(
        all_pair_scores,
        sample_count=sample_count,
        seed=bootstrap_seed,
    )
    truncation = _rate(total_truncated_games, total_games)
    reasons = _decision_reasons(
        anchor_results=anchor_results,
        overall=overall_posterior,
        truncation=truncation,
        stack=stack,
    )

    result = PromotionGateResult(
        focal_policy_id=focal_policy_id,
        ordered_opponents=tuple(anchor.name for anchor in anchors),
        record_path=_display_path(record_path, root=run_dir),
        seed_file_path=_display_path(seed_file, root=stack.root),
        seed_file_sha256=hash_seed_file(seed_file),
        paired_seed_count=len(paired_seeds),
        weighting=league.promotion_gate.weighting,
        seat_swap=league.promotion_gate.seat_swap,
        folding=league.promotion_gate.folding,
        anchors=tuple(anchor_results),
        overall_posterior=overall_posterior,
        truncation=truncation,
        passed=not reasons,
        reasons=tuple(reasons),
    )
    result.write_json(record_path)
    return result


def resolve_promotion_gate_anchors(
    stack: StackConfig,
    anchor_policy_ids: Mapping[str, str],
) -> tuple[PromotionGateAnchor, ...]:
    league = _require_league_config(stack)
    required = league.promotion_anchor_set_v1.required
    optional = league.promotion_anchor_set_v1.optional_if_available

    missing = [name for name in required if name not in anchor_policy_ids]
    if missing:
        raise ValueError(f"missing required promotion anchors: {', '.join(missing)}")

    anchors = [PromotionGateAnchor(name=name, policy_id=anchor_policy_ids[name]) for name in required]
    anchors.extend(
        PromotionGateAnchor(name=name, policy_id=anchor_policy_ids[name])
        for name in optional
        if name in anchor_policy_ids
    )
    return tuple(anchors)


def resolve_promotion_gate_seed_file(stack: StackConfig) -> Path:
    league = _require_league_config(stack)
    from_seed_sets = stack.seed_sets.get("promotion_gate")
    from_league = _resolve_repo_path(stack.root, league.promotion_seed_file)

    if from_seed_sets is None:
        return from_league
    if from_seed_sets != from_league:
        raise ValueError(
            "promotion gate seed file mismatch between stack.seed_sets['promotion_gate'] "
            f"({from_seed_sets}) and league.promotion_seed_file ({from_league})"
        )
    return from_seed_sets


def _build_anchor_result(
    *,
    anchor: PromotionGateAnchor,
    matchup: EvalRunResult,
    run_dir: Path,
    sample_count: int,
    bootstrap_seed: int | None,
) -> PromotionGateAnchorResult:
    pair_scores = paired_seed_scores(matchup.records, scheme="S0")
    posterior = PromotionGatePosterior.from_scores(
        pair_scores,
        sample_count=sample_count,
        seed=bootstrap_seed,
    )
    truncated_games = sum(1 for record in matchup.records if record.truncated)
    return PromotionGateAnchorResult(
        anchor_name=anchor.name,
        opponent_policy_id=anchor.policy_id,
        episodes_path=_display_path(matchup.episodes_path, root=run_dir),
        matchup_summary=matchup.summary,
        truncation=_rate(truncated_games, len(matchup.records)),
        posterior=posterior,
    )


def _decision_reasons(
    *,
    anchor_results: Sequence[PromotionGateAnchorResult],
    overall: PromotionGatePosterior,
    truncation: PromotionGateRate,
    stack: StackConfig,
) -> list[dict[str, Any]]:
    league = _require_league_config(stack)
    reasons: list[dict[str, Any]] = []

    if overall.prob_gt_target <= _PROMOTION_POSTERIOR_MIN:
        reasons.append(
            {
                "code": "overall_posterior_below_threshold",
                "observed": overall.prob_gt_target,
                "required_gt": _PROMOTION_POSTERIOR_MIN,
                "threshold": _PROMOTION_MEAN_THRESHOLD,
            }
        )

    max_anchor_loss_prob = league.promotion_gate.guardrails.max_prob_anchor_loss_below_0_45
    for anchor in anchor_results:
        if anchor.posterior.prob_lt_guardrail >= max_anchor_loss_prob:
            reasons.append(
                {
                    "anchor_name": anchor.anchor_name,
                    "code": "anchor_loss_guardrail_exceeded",
                    "max_allowed": max_anchor_loss_prob,
                    "observed": anchor.posterior.prob_lt_guardrail,
                    "threshold": _PROMOTION_LOSS_THRESHOLD,
                }
            )

    max_truncation_rate = league.promotion_gate.guardrails.max_truncation_rate
    if truncation.rate > max_truncation_rate:
        reasons.append(
            {
                "code": "truncation_rate_exceeded",
                "max_allowed": max_truncation_rate,
                "observed": truncation.rate,
                "truncated_games": truncation.numerator,
                "total_games": truncation.denominator,
            }
        )

    return reasons


def _posterior_means(scores: Sequence[float], *, sample_count: int, seed: int | None) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("promotion gate requires at least one paired-seed score")
    if not np.isfinite(score_array).all():
        raise ValueError("promotion gate scores must be finite")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    baseline = float(score_array[0])
    if score_array.size == 1 or np.all(score_array == baseline):
        return np.full((sample_count,), baseline, dtype=np.float64)
    rng = np.random.default_rng(seed)
    weights = rng.exponential(scale=1.0, size=(sample_count, score_array.size))
    weights /= np.sum(weights, axis=1, keepdims=True)
    centered = np.asarray(score_array - baseline, dtype=np.float64)
    return baseline + np.sum(weights * centered[np.newaxis, :], axis=1, dtype=np.float64)


def _rate(numerator: int, denominator: int) -> PromotionGateRate:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return PromotionGateRate(
        numerator=int(numerator),
        denominator=int(denominator),
        rate=float(numerator) / float(denominator),
    )


def _require_league_config(stack: StackConfig) -> LeagueConfig:
    league = stack.config.league
    if league is None:
        raise ValueError("stack.config.league is required for promotion gate")
    return league


def _validate_gate_contract(stack: StackConfig) -> None:
    league = _require_league_config(stack)
    gate = league.promotion_gate
    if gate.uncertainty_method != "bayesian_bootstrap_seedlevel_v1":
        raise ValueError(
            "promotion gate requires uncertainty_method='bayesian_bootstrap_seedlevel_v1', "
            f"got {gate.uncertainty_method!r}"
        )
    if gate.weighting != "uniform_across_anchors":
        raise ValueError(f"promotion gate requires weighting='uniform_across_anchors', got {gate.weighting!r}")
    if gate.folding != "S0":
        raise ValueError(f"promotion gate requires folding='S0', got {gate.folding!r}")
    if gate.seat_swap is not True:
        raise ValueError("promotion gate requires seat_swap=true")


def _validate_seed_count(paired_seeds: Sequence[int], *, expected: int, seed_file: Path) -> None:
    if len(paired_seeds) != expected:
        raise ValueError(f"promotion gate expected {expected} paired seeds in {seed_file}, found {len(paired_seeds)}")


def _resolve_repo_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "anchor"
