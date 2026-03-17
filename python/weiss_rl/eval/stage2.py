"""Stage-2 adaptive evaluation summaries on top of seat-swapped records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.harness import MatchupSummary, EvalGameRecord, summarize_game_records
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.uncertainty import EvalUncertaintySummary, paired_seed_uncertainty_summary

Stage2StopReason = Literal["continue", "decisive", "precision", "budget"]

__all__ = [
    "Stage2Decision",
    "Stage2StopReason",
    "summarize_stage2_records",
]


@dataclass(frozen=True, slots=True)
class Stage2Decision:
    summary: MatchupSummary
    uncertainty: EvalUncertaintySummary
    max_paired_seeds: int
    stop_reason: Stage2StopReason

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
    uncertainty = paired_seed_uncertainty_summary(
        records,
        scheme=scheme,
        sample_count=sample_count,
        ci_level=stop_rules.stop_confidence if ci_level is None else ci_level,
        seed=seed,
    )
    stop_reason = _stage2_stop_reason(
        uncertainty=uncertainty,
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
    )
    return Stage2Decision(
        summary=summary,
        uncertainty=uncertainty,
        max_paired_seeds=max_paired_seeds,
        stop_reason=stop_reason,
    )


def _stage2_stop_reason(
    *,
    uncertainty: EvalUncertaintySummary,
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
) -> Stage2StopReason:
    stop_confidence = float(stop_rules.stop_confidence)
    if uncertainty.prob_gt_half >= stop_confidence or uncertainty.prob_lt_half >= stop_confidence:
        return "decisive"
    if uncertainty.ci_half_width <= float(stop_rules.stop_delta_ci_half_width):
        return "precision"
    if uncertainty.paired_seed_count >= int(max_paired_seeds):
        return "budget"
    return "continue"
