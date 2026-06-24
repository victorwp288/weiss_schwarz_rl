"""Actor/model synchronization and off-policy sections for learning progress."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.progress.learning_progress_math import (
    _numeric_values,
    _paired_update_values,
    _pearson_correlation,
    _window_summary,
)

_STALE_POLICY_CORRELATION_KEYS = (
    "vtrace_rho_mean",
    "vtrace_rho_p99",
    "vtrace_train_rho_p95",
    "vtrace_train_rho_p99",
    "vtrace_clip_rate",
)


@dataclass(frozen=True, slots=True)
class ActorSyncSeries:
    policy_version_lag_p50_values: list[float]
    policy_version_lag_p90_values: list[float]
    learner_actor_update_lag_p50_values: list[float]
    learner_actor_update_lag_p90_values: list[float]
    league_update_lag_values: list[float]
    lag_warning_values: list[float]
    lag_warning_source: str
    stale_policy_pairs: dict[str, list[tuple[float, float]]]


@dataclass(frozen=True, slots=True)
class OffPolicySeries:
    vtrace_rho_values: list[float]
    vtrace_rho_p99_values: list[float]
    vtrace_train_rho_values: list[float]
    vtrace_train_rho_p95_values: list[float]
    vtrace_train_rho_p99_values: list[float]
    vtrace_clip_rate_values: list[float]
    logp_delta_abs_values: list[float]
    logp_delta_abs_p99_values: list[float]
    train_logp_delta_abs_values: list[float]
    train_logp_delta_abs_p99_values: list[float]


def build_actor_sync_series(
    *,
    records_for_route: Iterable[dict[str, Any]],
    metrics: Iterable[dict[str, Any]],
) -> ActorSyncSeries:
    route_records = list(records_for_route)
    metric_records = list(metrics)
    policy_version_lag_p50_values = _numeric_values(route_records, "policy_version_lag_p50")
    policy_version_lag_p90_values = _numeric_values(route_records, "policy_version_lag_p90")
    learner_actor_update_lag_p50_values = _numeric_values(route_records, "learner_actor_update_lag_p50")
    learner_actor_update_lag_p90_values = _numeric_values(route_records, "learner_actor_update_lag_p90")
    league_update_lag_values = _numeric_values(route_records, "league_update_lag")
    lag_warning_values = learner_actor_update_lag_p90_values or league_update_lag_values or policy_version_lag_p90_values
    if learner_actor_update_lag_p90_values:
        lag_warning_source = "learner_actor_update_lag_p90"
    elif league_update_lag_values:
        lag_warning_source = "league_update_lag"
    else:
        lag_warning_source = "policy_version_lag_p90"

    stale_policy_pairs = {
        key: _paired_update_values(route_records, lag_warning_source, metric_records, key)
        for key in _STALE_POLICY_CORRELATION_KEYS
    }
    return ActorSyncSeries(
        policy_version_lag_p50_values=policy_version_lag_p50_values,
        policy_version_lag_p90_values=policy_version_lag_p90_values,
        learner_actor_update_lag_p50_values=learner_actor_update_lag_p50_values,
        learner_actor_update_lag_p90_values=learner_actor_update_lag_p90_values,
        league_update_lag_values=league_update_lag_values,
        lag_warning_values=lag_warning_values,
        lag_warning_source=lag_warning_source,
        stale_policy_pairs=stale_policy_pairs,
    )


def build_off_policy_series(metrics: Iterable[dict[str, Any]]) -> OffPolicySeries:
    metric_records = list(metrics)
    return OffPolicySeries(
        vtrace_rho_values=_numeric_values(metric_records, "vtrace_rho_mean"),
        vtrace_rho_p99_values=_numeric_values(metric_records, "vtrace_rho_p99"),
        vtrace_train_rho_values=_numeric_values(metric_records, "vtrace_train_rho_mean"),
        vtrace_train_rho_p95_values=_numeric_values(metric_records, "vtrace_train_rho_p95"),
        vtrace_train_rho_p99_values=_numeric_values(metric_records, "vtrace_train_rho_p99"),
        vtrace_clip_rate_values=_numeric_values(metric_records, "vtrace_clip_rate"),
        logp_delta_abs_values=_numeric_values(metric_records, "target_behavior_logp_delta_abs_mean"),
        logp_delta_abs_p99_values=_numeric_values(metric_records, "target_behavior_logp_delta_abs_p99"),
        train_logp_delta_abs_values=_numeric_values(metric_records, "target_behavior_train_logp_delta_abs_mean"),
        train_logp_delta_abs_p99_values=_numeric_values(metric_records, "target_behavior_train_logp_delta_abs_p99"),
    )


def build_actor_model_sync_section(series: ActorSyncSeries) -> dict[str, Any]:
    return {
        "policy_version_lag_p50": _window_summary(series.policy_version_lag_p50_values, window=20),
        "policy_version_lag_p90": _window_summary(series.policy_version_lag_p90_values, window=20),
        "max_policy_version_lag_p90": None
        if not series.policy_version_lag_p90_values
        else max(series.policy_version_lag_p90_values),
        "learner_actor_update_lag_p50": _window_summary(series.learner_actor_update_lag_p50_values, window=20),
        "learner_actor_update_lag_p90": _window_summary(series.learner_actor_update_lag_p90_values, window=20),
        "max_learner_actor_update_lag_p90": None
        if not series.learner_actor_update_lag_p90_values
        else max(series.learner_actor_update_lag_p90_values),
        "lag_warning_source": series.lag_warning_source,
        "learner_to_actor_update_lag": _window_summary(series.lag_warning_values, window=20),
        "max_learner_to_actor_update_lag": None
        if not series.lag_warning_values
        else max(series.lag_warning_values),
    }


def build_league_sync_section(series: ActorSyncSeries) -> dict[str, Any]:
    return {
        "league_update_lag": _window_summary(series.league_update_lag_values, window=20),
        "max_league_update_lag": None if not series.league_update_lag_values else max(series.league_update_lag_values),
    }


def build_off_policy_section(
    *,
    actor_sync: ActorSyncSeries,
    off_policy: OffPolicySeries,
) -> dict[str, Any]:
    return {
        "vtrace_rho_mean": _window_summary(off_policy.vtrace_rho_values, window=20),
        "vtrace_rho_p99": _window_summary(off_policy.vtrace_rho_p99_values, window=20),
        "vtrace_train_rho_mean": _window_summary(off_policy.vtrace_train_rho_values, window=20),
        "vtrace_train_rho_p95": _window_summary(off_policy.vtrace_train_rho_p95_values, window=20),
        "vtrace_train_rho_p99": _window_summary(off_policy.vtrace_train_rho_p99_values, window=20),
        "vtrace_clip_rate": _window_summary(off_policy.vtrace_clip_rate_values, window=20),
        "target_behavior_logp_delta_abs_mean": _window_summary(off_policy.logp_delta_abs_values, window=20),
        "target_behavior_logp_delta_abs_p99": _window_summary(off_policy.logp_delta_abs_p99_values, window=20),
        "target_behavior_train_logp_delta_abs_mean": _window_summary(
            off_policy.train_logp_delta_abs_values,
            window=20,
        ),
        "target_behavior_train_logp_delta_abs_p99": _window_summary(
            off_policy.train_logp_delta_abs_p99_values,
            window=20,
        ),
        "max_vtrace_rho_mean": None if not off_policy.vtrace_rho_values else max(off_policy.vtrace_rho_values),
        "max_vtrace_rho_p99": None if not off_policy.vtrace_rho_p99_values else max(off_policy.vtrace_rho_p99_values),
        "max_vtrace_train_rho_mean": None
        if not off_policy.vtrace_train_rho_values
        else max(off_policy.vtrace_train_rho_values),
        "max_vtrace_train_rho_p95": None
        if not off_policy.vtrace_train_rho_p95_values
        else max(off_policy.vtrace_train_rho_p95_values),
        "max_vtrace_train_rho_p99": None
        if not off_policy.vtrace_train_rho_p99_values
        else max(off_policy.vtrace_train_rho_p99_values),
        "max_vtrace_clip_rate": None
        if not off_policy.vtrace_clip_rate_values
        else max(off_policy.vtrace_clip_rate_values),
        "max_target_behavior_logp_delta_abs_mean": None
        if not off_policy.logp_delta_abs_values
        else max(off_policy.logp_delta_abs_values),
        "max_target_behavior_logp_delta_abs_p99": None
        if not off_policy.logp_delta_abs_p99_values
        else max(off_policy.logp_delta_abs_p99_values),
        "max_target_behavior_train_logp_delta_abs_mean": None
        if not off_policy.train_logp_delta_abs_values
        else max(off_policy.train_logp_delta_abs_values),
        "max_target_behavior_train_logp_delta_abs_p99": None
        if not off_policy.train_logp_delta_abs_p99_values
        else max(off_policy.train_logp_delta_abs_p99_values),
        "stale_policy_lag_source": actor_sync.lag_warning_source,
        "stale_policy_lag_correlations": {
            key: {
                "paired_update_count": len(pairs),
                "pearson": _pearson_correlation(pairs),
            }
            for key, pairs in actor_sync.stale_policy_pairs.items()
        },
    }
