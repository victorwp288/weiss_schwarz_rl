"""Periodic dev-eval summary and stall-monitor persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from weiss_rl.training.dev_eval.common import (
    DevEvalTrainingPaths,
    load_json_object,
    periodic_dev_eval_summaries_path,
    stall_monitor_state_path,
    write_json,
)


def summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def persist_periodic_dev_eval_summary(
    *,
    training_paths: DevEvalTrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = periodic_dev_eval_summaries_path(training_paths)
    summaries = load_json_object(path, label="periodic dev-eval summaries") if path.is_file() else {}
    summaries[focal_policy_id] = {
        "aggregate_score": float(payload.get("aggregate_score", 0.0)),
        "anchor_scores": dict(cast(Mapping[str, Any], payload.get("anchor_scores", {}))),
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
    }
    write_json(path, summaries)


def update_stall_monitor(
    *,
    stack: Any,
    training_paths: DevEvalTrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.stall_monitor.enabled:
        return None
    threshold = float(curriculum.stall_monitor.truncation_rate_threshold)
    required_consecutive = int(curriculum.stall_monitor.consecutive_evals)
    anchors_raw = summary_payload.get("anchors", {})
    if not isinstance(anchors_raw, Mapping):
        return None

    anchor_truncation_rates: dict[str, float] = {}
    anchor_no_progress_rates: dict[str, float] = {}
    anchor_natural_timeout_rates: dict[str, float] = {}
    anchor_stall_rates: dict[str, float] = {}
    for anchor_name, anchor_payload in anchors_raw.items():
        if not isinstance(anchor_payload, Mapping):
            continue
        matchup_summary = anchor_payload.get("summary", {})
        if not isinstance(matchup_summary, Mapping):
            continue
        truncation_rate = summary_rate(matchup_summary, "truncations")
        no_progress_rate = summary_rate(matchup_summary, "no_progress_timeouts")
        natural_timeout_rate = summary_rate(matchup_summary, "natural_timeouts")
        if truncation_rate is None and no_progress_rate is None and natural_timeout_rate is None:
            continue
        anchor_truncation_rates[str(anchor_name)] = 0.0 if truncation_rate is None else truncation_rate
        anchor_no_progress_rates[str(anchor_name)] = 0.0 if no_progress_rate is None else no_progress_rate
        anchor_natural_timeout_rates[str(anchor_name)] = 0.0 if natural_timeout_rate is None else natural_timeout_rate
        anchor_stall_rates[str(anchor_name)] = (
            anchor_no_progress_rates[str(anchor_name)]
            if no_progress_rate is not None
            else anchor_truncation_rates[str(anchor_name)]
        )
    if not anchor_stall_rates:
        return None

    state_path = stall_monitor_state_path(training_paths)
    state = load_json_object(state_path, label="stall monitor state") if state_path.is_file() else {}
    previous_consecutive = int(state.get("consecutive_trigger_count", 0))
    worst_anchor = max(anchor_stall_rates, key=lambda anchor_name: anchor_stall_rates[anchor_name])
    worst_rate = float(anchor_stall_rates[worst_anchor])
    consecutive = previous_consecutive + 1 if worst_rate >= threshold else 0
    stall_risk = consecutive >= required_consecutive
    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "threshold": threshold,
        "required_consecutive_evals": required_consecutive,
        "consecutive_trigger_count": consecutive,
        "stall_risk": stall_risk,
        "worst_anchor": worst_anchor,
        "stall_indicator_kind": (
            "no_progress_timeout" if anchor_no_progress_rates.get(worst_anchor, 0.0) > 0.0 else "truncation_fallback"
        ),
        "worst_stall_rate": worst_rate,
        "worst_truncation_rate": float(anchor_truncation_rates.get(worst_anchor, 0.0)),
        "worst_no_progress_timeout_rate": float(anchor_no_progress_rates.get(worst_anchor, 0.0)),
        "worst_natural_timeout_rate": float(anchor_natural_timeout_rates.get(worst_anchor, 0.0)),
        "anchor_truncation_rates": anchor_truncation_rates,
        "anchor_no_progress_timeout_rates": anchor_no_progress_rates,
        "anchor_natural_timeout_rates": anchor_natural_timeout_rates,
    }
    write_json(state_path, payload)
    return payload


__all__ = [
    "persist_periodic_dev_eval_summary",
    "summary_rate",
    "update_stall_monitor",
]
