from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BestDevEvalCheckpoint:
    score: float
    update_count: int


def best_dev_eval_checkpoint(
    record: Mapping[str, Any] | None,
    *,
    learner_update_count: int | None = None,
    require_prior_update: bool = False,
) -> BestDevEvalCheckpoint | None:
    if record is None:
        return None
    metric_kind = str(record.get("metric_kind", "")).strip()
    metric_value = record.get("metric_value")
    update_count = record.get("update_count")
    if metric_kind != "dev_eval_mean":
        return None
    if not isinstance(metric_value, (int, float)) or not math.isfinite(float(metric_value)):
        return None
    if not isinstance(update_count, int):
        return None
    if require_prior_update:
        if learner_update_count is None:
            raise ValueError("learner_update_count is required when require_prior_update is true")
        if int(update_count) >= int(learner_update_count):
            return None
    return BestDevEvalCheckpoint(score=float(metric_value), update_count=int(update_count))


def checkpoint_guard_rollback_reasons(
    *,
    checkpoint_guard: Any,
    current_score: float,
    best_score: float,
    worst_stall_rate: float | None,
    max_prob_lt_half: float | None,
) -> list[str]:
    reasons: list[str] = []
    if float(current_score) <= float(best_score) - float(checkpoint_guard.rollback_score_margin):
        reasons.append("score_drop")
    if worst_stall_rate is not None and (
        float(worst_stall_rate) >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        reasons.append("truncation")
    if max_prob_lt_half is not None and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half)):
        reasons.append("confidence")
    return reasons


def build_rollback_to_best_event_payload(
    *,
    learner_update_count: int,
    policy_version: int,
    current_score: float,
    best: BestDevEvalCheckpoint,
    worst_stall_rate: float | None,
    worst_truncation_rate: float | None,
    worst_no_progress_timeout_rate: float | None,
    worst_natural_timeout_rate: float | None,
    confidence: Mapping[str, float | None],
    reasons: Sequence[str],
    best_checkpoint_path: str,
    latest_checkpoint_path: str,
    publish_metrics: Mapping[str, Any],
    latest_metrics: Mapping[str, float] | None,
    demoted_champions: Sequence[str],
) -> dict[str, Any]:
    return {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner_update_count),
        "policy_version": int(policy_version),
        "current_score": float(current_score),
        "best_score": float(best.score),
        "best_update_count": int(best.update_count),
        "worst_stall_rate": worst_stall_rate,
        "worst_truncation_rate": worst_truncation_rate,
        "worst_no_progress_timeout_rate": worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": worst_natural_timeout_rate,
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "reasons": list(reasons),
        "best_checkpoint_path": best_checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "rolled_back_checkpoint_path": best_checkpoint_path,
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": list(demoted_champions),
    }


def build_finalize_to_best_event_payload(
    *,
    learner_update_count: int,
    policy_version: int,
    current_score: float,
    best: BestDevEvalCheckpoint,
    confidence: Mapping[str, float | None],
    latest_metrics: Mapping[str, float] | None,
    best_checkpoint_path: str | Path,
    latest_checkpoint_path: str | Path,
    demoted_champions: Sequence[str],
) -> dict[str, Any]:
    return {
        "format": "checkpoint_guard_event_v1",
        "action": "finalize_to_best",
        "update_count": int(learner_update_count),
        "policy_version": int(policy_version),
        "current_score": float(current_score),
        "best_score": float(best.score),
        "best_update_count": int(best.update_count),
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "demoted_champions": list(demoted_champions),
    }
