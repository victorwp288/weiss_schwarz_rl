"""Checkpoint alias candidate scoring and diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.training.checkpointing.guard import (
    checkpoint_candidate_metric,
    dev_eval_aggregate_score,
    dev_eval_confidence_stats,
    dev_eval_ineligibility_reasons,
    dev_eval_worst_natural_timeout_rate,
    dev_eval_worst_no_progress_timeout_rate,
    dev_eval_worst_stall_rate,
    dev_eval_worst_truncation_rate,
)


@dataclass(frozen=True, slots=True)
class CheckpointAliasCandidate:
    metric_kind: str | None
    metric_value: float | None
    observed_score: float | None
    dev_eval_candidate: dict[str, Any] | None


def dev_eval_candidate_diagnostics(
    *,
    stack: Any,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if dev_eval_summary is None:
        return None
    reasons = dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    confidence = dev_eval_confidence_stats(dev_eval_summary)
    return {
        "score": dev_eval_aggregate_score(dev_eval_summary),
        "eligible_for_best": not reasons,
        "ineligibility_reasons": list(reasons),
        "confidence": confidence,
        "worst_truncation_rate": dev_eval_worst_truncation_rate(dev_eval_summary),
        "worst_no_progress_timeout_rate": dev_eval_worst_no_progress_timeout_rate(dev_eval_summary),
        "worst_natural_timeout_rate": dev_eval_worst_natural_timeout_rate(dev_eval_summary),
        "worst_stall_rate": dev_eval_worst_stall_rate(dev_eval_summary),
    }


def checkpoint_alias_candidate(
    *,
    stack: Any,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> CheckpointAliasCandidate:
    metric_kind, metric_value = checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    return CheckpointAliasCandidate(
        metric_kind=metric_kind,
        metric_value=metric_value,
        observed_score=dev_eval_aggregate_score(dev_eval_summary),
        dev_eval_candidate=dev_eval_candidate_diagnostics(stack=stack, dev_eval_summary=dev_eval_summary),
    )


def should_update_observed_best(
    *,
    existing_record: Mapping[str, Any] | None,
    observed_score: float | None,
) -> bool:
    if observed_score is None:
        return False
    observed_best_value = None if existing_record is None else existing_record.get("metric_value")
    return (
        not isinstance(observed_best_value, (int, float))
        or not math.isfinite(float(observed_best_value))
        or float(observed_score) > float(observed_best_value)
    )


__all__ = [
    "CheckpointAliasCandidate",
    "checkpoint_alias_candidate",
    "dev_eval_candidate_diagnostics",
    "should_update_observed_best",
]
