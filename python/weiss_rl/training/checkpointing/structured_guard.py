"""Structured-mainmove checkpoint guard warning logic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from weiss_rl.training.checkpointing.aliases import LearnerRecordSource
from weiss_rl.training.checkpointing.guard import dev_eval_aggregate_score


def extract_structured_guard_b2_anchor_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return None
    for key, value in anchor_scores.items():
        key_text = str(key).strip().lower()
        if "b2" not in key_text:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def structured_mainmove_guard_warning_payload(
    *,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if latest_metrics is None:
        return None
    top1_rate = latest_metrics.get("structured_main_move_0_2_top1_rate")
    move_share = latest_metrics.get("structured_main_move_share_when_play_available")
    if top1_rate is None or move_share is None:
        return None
    if not math.isfinite(float(top1_rate)) or not math.isfinite(float(move_share)):
        return None
    if float(top1_rate) < 0.15 and float(move_share) < 0.35:
        return None

    aggregate_score = dev_eval_aggregate_score(dev_eval_summary) if dev_eval_summary is not None else None
    b2_score = extract_structured_guard_b2_anchor_score(dev_eval_summary)
    if b2_score is not None and float(b2_score) > 0.10:
        return None
    if b2_score is None and aggregate_score is not None and float(aggregate_score) > 0.40:
        return None

    return {
        "format": "checkpoint_guard_event_v1",
        "event_kind": "structured_mainmove_warning_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "structured_main_move_0_2_top1_rate": float(top1_rate),
        "structured_main_move_share_when_play_available": float(move_share),
        "dev_eval_aggregate_score": None if aggregate_score is None else float(aggregate_score),
        "b2_anchor_score": None if b2_score is None else float(b2_score),
    }


__all__ = [
    "extract_structured_guard_b2_anchor_score",
    "structured_mainmove_guard_warning_payload",
]
