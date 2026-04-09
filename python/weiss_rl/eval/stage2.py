"""Stage-2 adaptive evaluation summaries on top of seat-swapped records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.harness import EvalGameRecord, MatchupSummary, summarize_game_records
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores
from weiss_rl.eval.uncertainty import EvalUncertaintySummary, bayesian_bootstrap_summary

Stage2StopReason = Literal["continue", "decisive", "precision", "budget", "no_included_pairs"]

__all__ = [
    "Stage2Decision",
    "Stage2StopReason",
    "summarize_stage2_records",
]


@dataclass(frozen=True, slots=True)
class Stage2Decision:
    summary: MatchupSummary
    uncertainty: EvalUncertaintySummary | None
    max_paired_seeds: int
    observed_paired_seeds: int
    excluded_paired_seeds: int
    stop_reason: Stage2StopReason

    @property
    def paired_seed_count(self) -> int:
        if self.uncertainty is None:
            return 0
        return self.uncertainty.paired_seed_count

    @property
    def has_payoff_samples(self) -> bool:
        return self.uncertainty is not None

    @property
    def should_stop(self) -> bool:
        return self.stop_reason != "continue"


def summarize_stage2_records(
    records: list[EvalGameRecord] | tuple[EvalGameRecord, ...],
    *,
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
    scheme: PayoffFoldScheme = "S0",
    sample_count: int = 1000,
    ci_level: float | None = None,
    seed: int | None = None,
) -> Stage2Decision:
    if max_paired_seeds <= 0:
        raise ValueError("max_paired_seeds must be positive")

    summary = summarize_game_records(records)
    pair_scores = paired_seed_scores(records, scheme=scheme)
    observed_paired_seeds = len({int(record.pair_index) for record in records})
    excluded_paired_seeds = observed_paired_seeds - len(pair_scores)
    uncertainty = None
    if pair_scores:
        uncertainty = bayesian_bootstrap_summary(
            pair_scores,
            sample_count=sample_count,
            ci_level=stop_rules.stop_confidence if ci_level is None else ci_level,
            seed=seed,
        )
    stop_reason = _stage2_stop_reason(
        uncertainty=uncertainty,
        observed_paired_seeds=observed_paired_seeds,
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
    )
    return Stage2Decision(
        summary=summary,
        uncertainty=uncertainty,
        max_paired_seeds=max_paired_seeds,
        observed_paired_seeds=observed_paired_seeds,
        excluded_paired_seeds=excluded_paired_seeds,
        stop_reason=stop_reason,
    )


def _stage2_stop_reason(
    *,
    uncertainty: EvalUncertaintySummary | None,
    observed_paired_seeds: int,
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
) -> Stage2StopReason:
    if uncertainty is not None:
        stop_confidence = float(stop_rules.stop_confidence)
        if uncertainty.prob_gt_half >= stop_confidence or uncertainty.prob_lt_half >= stop_confidence:
            return "decisive"
        if uncertainty.ci_half_width <= float(stop_rules.stop_delta_ci_half_width):
            return "precision"
    if observed_paired_seeds >= int(max_paired_seeds):
        if uncertainty is None:
            return "no_included_pairs"
        return "budget"
    return "continue"
