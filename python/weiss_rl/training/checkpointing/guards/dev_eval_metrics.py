"""Read scalar checkpoint metrics from periodic dev-eval summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DevEvalConfidenceStats:
    min_prob_gt_half: float | None
    max_prob_lt_half: float | None
    max_ci_half_width: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "min_prob_gt_half": self.min_prob_gt_half,
            "max_prob_lt_half": self.max_prob_lt_half,
            "max_ci_half_width": self.max_ci_half_width,
        }


@dataclass(frozen=True, slots=True)
class DevEvalTimeoutRates:
    worst_truncation_rate: float | None
    worst_no_progress_timeout_rate: float | None
    worst_natural_timeout_rate: float | None

    @property
    def worst_stall_rate(self) -> float | None:
        if self.worst_no_progress_timeout_rate is not None:
            return self.worst_no_progress_timeout_rate
        return self.worst_truncation_rate


def dev_eval_aggregate_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    aggregate_score = dev_eval_summary.get("aggregate_score")
    if isinstance(aggregate_score, (int, float)) and np.isfinite(float(aggregate_score)):
        return float(aggregate_score)
    uncertainty = dev_eval_summary.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        mean_value = uncertainty.get("mean")
        if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
            return float(mean_value)
    return None


def summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def dev_eval_worst_truncation_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        monitor_worst_rate = stall_monitor.get("worst_truncation_rate")
        if isinstance(monitor_worst_rate, (int, float)) and np.isfinite(float(monitor_worst_rate)):
            return float(monitor_worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        rate = summary_rate(summary, "truncations")
        if rate is None:
            continue
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def dev_eval_worst_reason_rate(
    dev_eval_summary: Mapping[str, Any] | None,
    *,
    summary_key: str,
    stall_monitor_key: str,
) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        monitor_worst_rate = stall_monitor.get(stall_monitor_key)
        if isinstance(monitor_worst_rate, (int, float)) and np.isfinite(float(monitor_worst_rate)):
            return float(monitor_worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        rate = summary_rate(summary, summary_key)
        if rate is None:
            continue
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def dev_eval_worst_no_progress_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="no_progress_timeouts",
        stall_monitor_key="worst_no_progress_timeout_rate",
    )


def dev_eval_worst_natural_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="natural_timeouts",
        stall_monitor_key="worst_natural_timeout_rate",
    )


def collect_dev_eval_timeout_rates(dev_eval_summary: Mapping[str, Any] | None) -> DevEvalTimeoutRates:
    return DevEvalTimeoutRates(
        worst_truncation_rate=dev_eval_worst_truncation_rate(dev_eval_summary),
        worst_no_progress_timeout_rate=dev_eval_worst_no_progress_timeout_rate(dev_eval_summary),
        worst_natural_timeout_rate=dev_eval_worst_natural_timeout_rate(dev_eval_summary),
    )


def dev_eval_worst_stall_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return collect_dev_eval_timeout_rates(dev_eval_summary).worst_stall_rate


def collect_dev_eval_confidence_stats(dev_eval_summary: Mapping[str, Any] | None) -> DevEvalConfidenceStats:
    if dev_eval_summary is None:
        return DevEvalConfidenceStats(
            min_prob_gt_half=None,
            max_prob_lt_half=None,
            max_ci_half_width=None,
        )
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return DevEvalConfidenceStats(
            min_prob_gt_half=None,
            max_prob_lt_half=None,
            max_ci_half_width=None,
        )
    min_prob_gt_half: float | None = None
    max_prob_lt_half: float | None = None
    max_ci_half_width: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        uncertainty = anchor_payload.get("uncertainty")
        if not isinstance(uncertainty, Mapping):
            continue
        prob_gt_half = uncertainty.get("prob_gt_half")
        prob_lt_half = uncertainty.get("prob_lt_half")
        ci_half_width = uncertainty.get("ci_half_width")
        if isinstance(prob_gt_half, (int, float)) and np.isfinite(float(prob_gt_half)):
            min_prob_gt_half = (
                float(prob_gt_half) if min_prob_gt_half is None else min(min_prob_gt_half, float(prob_gt_half))
            )
        if isinstance(prob_lt_half, (int, float)) and np.isfinite(float(prob_lt_half)):
            max_prob_lt_half = (
                float(prob_lt_half) if max_prob_lt_half is None else max(max_prob_lt_half, float(prob_lt_half))
            )
        if isinstance(ci_half_width, (int, float)) and np.isfinite(float(ci_half_width)):
            max_ci_half_width = (
                float(ci_half_width) if max_ci_half_width is None else max(max_ci_half_width, float(ci_half_width))
            )
    return DevEvalConfidenceStats(
        min_prob_gt_half=min_prob_gt_half,
        max_prob_lt_half=max_prob_lt_half,
        max_ci_half_width=max_ci_half_width,
    )


def dev_eval_confidence_stats(dev_eval_summary: Mapping[str, Any] | None) -> dict[str, float | None]:
    return collect_dev_eval_confidence_stats(dev_eval_summary).as_dict()


def dev_eval_worst_anchor_mean(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_mean: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        uncertainty = anchor_payload.get("uncertainty")
        if not isinstance(uncertainty, Mapping):
            continue
        mean_value = uncertainty.get("mean")
        if not isinstance(mean_value, (int, float)) or not np.isfinite(float(mean_value)):
            continue
        worst_mean = float(mean_value) if worst_mean is None else min(worst_mean, float(mean_value))
    return worst_mean


__all__ = [
    "DevEvalConfidenceStats",
    "DevEvalTimeoutRates",
    "collect_dev_eval_confidence_stats",
    "collect_dev_eval_timeout_rates",
    "dev_eval_aggregate_score",
    "dev_eval_confidence_stats",
    "dev_eval_worst_anchor_mean",
    "dev_eval_worst_natural_timeout_rate",
    "dev_eval_worst_no_progress_timeout_rate",
    "dev_eval_worst_reason_rate",
    "dev_eval_worst_stall_rate",
    "dev_eval_worst_truncation_rate",
    "summary_rate",
]
