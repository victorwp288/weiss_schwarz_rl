"""Checkpoint-guard rollback decision helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_aggregate_score,
    dev_eval_confidence_stats,
    dev_eval_is_authoritative,
    dev_eval_worst_natural_timeout_rate,
    dev_eval_worst_no_progress_timeout_rate,
    dev_eval_worst_stall_rate,
    dev_eval_worst_truncation_rate,
)


@dataclass(frozen=True, slots=True)
class CheckpointGuardRollbackPlan:
    current_score: float
    best_score: float
    best_update_count: int
    worst_stall_rate: float | None
    worst_truncation_rate: float | None
    worst_no_progress_timeout_rate: float | None
    worst_natural_timeout_rate: float | None
    min_prob_gt_half: float | None
    max_prob_lt_half: float | None
    max_ci_half_width: float | None
    reasons: tuple[str, ...]


def checkpoint_guard_rollback_plan(
    *,
    stack: StackConfig,
    learner_update_count: int,
    last_rollback_update: int | None,
    best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> CheckpointGuardRollbackPlan | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if not dev_eval_is_authoritative(dev_eval_summary):
        return None
    if last_rollback_update is not None and (int(learner_update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    if best_record is None:
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not math.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int) or int(best_update_count) >= int(learner_update_count):
        return None
    best_score = float(best_metric_value)
    if best_score < float(checkpoint_guard.min_best_score):
        return None

    worst_truncation_rate = dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    confidence = dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons: list[str] = []
    if current_score <= best_score - float(checkpoint_guard.rollback_score_margin):
        rollback_reasons.append("score_drop")
    if worst_stall_rate is not None and (
        worst_stall_rate >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        rollback_reasons.append("truncation")
    max_prob_lt_half = confidence["max_prob_lt_half"]
    if (
        current_score < best_score
        and max_prob_lt_half is not None
        and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half))
    ):
        rollback_reasons.append("confidence")
    if not rollback_reasons:
        return None

    return CheckpointGuardRollbackPlan(
        current_score=float(current_score),
        best_score=best_score,
        best_update_count=int(best_update_count),
        worst_stall_rate=worst_stall_rate,
        worst_truncation_rate=worst_truncation_rate,
        worst_no_progress_timeout_rate=worst_no_progress_timeout_rate,
        worst_natural_timeout_rate=worst_natural_timeout_rate,
        min_prob_gt_half=confidence["min_prob_gt_half"],
        max_prob_lt_half=confidence["max_prob_lt_half"],
        max_ci_half_width=confidence["max_ci_half_width"],
        reasons=tuple(rollback_reasons),
    )


def format_checkpoint_guard_rollback_message(guard_event: Mapping[str, Any]) -> str:
    return (
        "Checkpoint guard rollback: "
        f"update={guard_event['update_count']} "
        f"best_update={guard_event['best_update_count']} "
        f"current_score={float(guard_event['current_score']):.4f} "
        f"best_score={float(guard_event['best_score']):.4f} "
        f"reasons={','.join(str(reason) for reason in guard_event['reasons'])}"
    )


def format_checkpoint_guard_final_selection_message(guard_event: Mapping[str, Any]) -> str:
    return (
        "Checkpoint guard final selection: "
        f"update={guard_event['update_count']} "
        f"best_update={guard_event['best_update_count']} "
        f"current_score={float(guard_event['current_score']):.4f} "
        f"best_score={float(guard_event['best_score']):.4f}"
    )
