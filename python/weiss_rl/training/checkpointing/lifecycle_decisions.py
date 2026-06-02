"""Checkpoint guard rollback/finalize decisions and event payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from weiss_rl.training.checkpointing.guard import (
    dev_eval_aggregate_score,
    dev_eval_confidence_stats,
    dev_eval_worst_natural_timeout_rate,
    dev_eval_worst_no_progress_timeout_rate,
    dev_eval_worst_stall_rate,
    dev_eval_worst_truncation_rate,
)
from weiss_rl.training.checkpointing.guard_events import (
    BestDevEvalCheckpoint,
    best_dev_eval_checkpoint,
    build_finalize_to_best_event_payload,
    build_rollback_to_best_event_payload,
    checkpoint_guard_rollback_reasons,
)


@dataclass(frozen=True, slots=True)
class RollbackToBestDecision:
    current_score: float
    best: BestDevEvalCheckpoint
    confidence: Mapping[str, float | None]
    reasons: list[str]
    worst_truncation_rate: float | None
    worst_stall_rate: float | None
    worst_no_progress_timeout_rate: float | None
    worst_natural_timeout_rate: float | None


@dataclass(frozen=True, slots=True)
class FinalizeToBestDecision:
    current_score: float
    best: BestDevEvalCheckpoint
    confidence: Mapping[str, float | None]


def rollback_to_best_decision(
    *,
    checkpoint_guard: Any,
    best_record: object,
    learner_update_count: int,
    dev_eval_summary: Mapping[str, Any],
) -> RollbackToBestDecision | None:
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    worst_truncation_rate = dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    if not isinstance(best_record, Mapping):
        return None
    best_candidate = best_dev_eval_checkpoint(
        best_record,
        learner_update_count=learner_update_count,
        require_prior_update=True,
    )
    if best_candidate is None:
        return None
    if best_candidate.score < float(checkpoint_guard.min_best_score):
        return None

    confidence = dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons = checkpoint_guard_rollback_reasons(
        checkpoint_guard=checkpoint_guard,
        current_score=float(current_score),
        best_score=best_candidate.score,
        worst_stall_rate=worst_stall_rate,
        max_prob_lt_half=confidence["max_prob_lt_half"],
    )
    if not rollback_reasons:
        return None

    return RollbackToBestDecision(
        current_score=float(current_score),
        best=best_candidate,
        confidence=confidence,
        reasons=rollback_reasons,
        worst_truncation_rate=worst_truncation_rate,
        worst_stall_rate=worst_stall_rate,
        worst_no_progress_timeout_rate=worst_no_progress_timeout_rate,
        worst_natural_timeout_rate=worst_natural_timeout_rate,
    )


def rollback_to_best_event_payload(
    *,
    learner_update_count: int,
    policy_version: int,
    decision: RollbackToBestDecision,
    best_checkpoint_path: str,
    latest_checkpoint_path: str,
    publish_metrics: Mapping[str, Any],
    latest_metrics: Mapping[str, float] | None,
    demoted_champions: Sequence[str],
) -> dict[str, Any]:
    return build_rollback_to_best_event_payload(
        learner_update_count=learner_update_count,
        policy_version=policy_version,
        current_score=decision.current_score,
        best=decision.best,
        worst_stall_rate=decision.worst_stall_rate,
        worst_truncation_rate=decision.worst_truncation_rate,
        worst_no_progress_timeout_rate=decision.worst_no_progress_timeout_rate,
        worst_natural_timeout_rate=decision.worst_natural_timeout_rate,
        confidence=decision.confidence,
        reasons=decision.reasons,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        publish_metrics=publish_metrics,
        latest_metrics=latest_metrics,
        demoted_champions=demoted_champions,
    )


def finalize_to_best_decision(
    *,
    best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> FinalizeToBestDecision | None:
    if best_record is None:
        return None
    best_candidate = best_dev_eval_checkpoint(best_record)
    if best_candidate is None:
        return None
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None or current_score >= best_candidate.score:
        return None
    return FinalizeToBestDecision(
        current_score=float(current_score),
        best=best_candidate,
        confidence=dev_eval_confidence_stats(dev_eval_summary),
    )


def finalize_to_best_event_payload(
    *,
    learner_update_count: int,
    policy_version: int,
    decision: FinalizeToBestDecision,
    latest_metrics: Mapping[str, float] | None,
    best_checkpoint_path: str,
    latest_checkpoint_path: str,
    demoted_champions: Sequence[str],
) -> dict[str, Any]:
    return build_finalize_to_best_event_payload(
        learner_update_count=learner_update_count,
        policy_version=policy_version,
        current_score=decision.current_score,
        best=decision.best,
        confidence=decision.confidence,
        latest_metrics=latest_metrics,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        demoted_champions=demoted_champions,
    )


__all__ = [
    "BestDevEvalCheckpoint",
    "FinalizeToBestDecision",
    "RollbackToBestDecision",
    "finalize_to_best_decision",
    "finalize_to_best_event_payload",
    "rollback_to_best_decision",
    "rollback_to_best_event_payload",
]
