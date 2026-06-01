"""League guard rules for learning-progress diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_LEAGUE_GUARD_ANCHORS = (
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _latest_periodic_anchor_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    periodic = summary.get("periodic_dev_eval")
    if not isinstance(periodic, Mapping):
        return {}
    records = periodic.get("records")
    if not isinstance(records, list) or not records:
        return {}
    latest_record = records[-1]
    if not isinstance(latest_record, Mapping):
        return {}
    anchor_scores = latest_record.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in anchor_scores.items():
        score = _finite_float(value)
        if score is not None:
            result[str(key)] = score
    return result


def evaluate_league_guard(
    summary: Mapping[str, Any],
    *,
    required_anchors: Iterable[str] = DEFAULT_LEAGUE_GUARD_ANCHORS,
    min_latest_anchor_score: float | None = 0.45,
    max_latest_drop: float | None = 0.05,
    require_promotion_pass_after_attempts: int | None = 3,
    max_consecutive_promotion_failures: int | None = 3,
    max_vtrace_rho_p99: float | None = None,
) -> dict[str, Any]:
    """Evaluate machine-actionable gates for guarded league probes."""

    failures: list[dict[str, Any]] = []
    required_anchor_tuple = tuple(str(anchor) for anchor in required_anchors)
    anchor_scores = _latest_periodic_anchor_scores(summary)
    if min_latest_anchor_score is not None:
        for anchor in required_anchor_tuple:
            score = anchor_scores.get(anchor)
            if score is None:
                failures.append({"code": "missing_latest_anchor_score", "anchor": anchor})
            elif score < float(min_latest_anchor_score):
                failures.append(
                    {
                        "code": "latest_anchor_below_threshold",
                        "anchor": anchor,
                        "observed": score,
                        "threshold": float(min_latest_anchor_score),
                    }
                )
    periodic = summary.get("periodic_dev_eval")
    latest_minus_best = periodic.get("latest_minus_best") if isinstance(periodic, Mapping) else None
    latest_drop = _finite_float(latest_minus_best)
    if max_latest_drop is not None and latest_drop is not None and latest_drop < -float(max_latest_drop):
        failures.append(
            {
                "code": "latest_periodic_drop_exceeded",
                "observed": latest_drop,
                "threshold": -float(max_latest_drop),
            }
        )
    promotion_gate = summary.get("promotion_gate")
    if isinstance(promotion_gate, Mapping):
        attempt_count = promotion_gate.get("attempt_count")
        passed_count = promotion_gate.get("passed_count")
        consecutive_failure_count = promotion_gate.get("consecutive_failure_count")
        if (
            require_promotion_pass_after_attempts is not None
            and isinstance(attempt_count, int)
            and isinstance(passed_count, int)
            and attempt_count >= int(require_promotion_pass_after_attempts)
            and passed_count <= 0
        ):
            failures.append(
                {
                    "code": "promotion_gate_no_pass_after_attempts",
                    "attempt_count": attempt_count,
                    "passed_count": passed_count,
                    "threshold": int(require_promotion_pass_after_attempts),
                }
            )
        if (
            max_consecutive_promotion_failures is not None
            and isinstance(consecutive_failure_count, int)
            and consecutive_failure_count >= int(max_consecutive_promotion_failures)
        ):
            failures.append(
                {
                    "code": "promotion_gate_consecutive_failures_exceeded",
                    "observed": consecutive_failure_count,
                    "threshold": int(max_consecutive_promotion_failures),
                }
            )
    off_policy = summary.get("off_policy")
    max_rho_p99 = off_policy.get("max_vtrace_rho_p99") if isinstance(off_policy, Mapping) else None
    max_train_rho_p99 = off_policy.get("max_vtrace_train_rho_p99") if isinstance(off_policy, Mapping) else None
    max_train_rho_p95 = off_policy.get("max_vtrace_train_rho_p95") if isinstance(off_policy, Mapping) else None
    max_rho_p99_float = _finite_float(max_rho_p99)
    max_train_tail_float = _finite_float(max_train_rho_p99)
    if max_train_tail_float is None:
        max_train_tail_float = _finite_float(max_train_rho_p95)
    guard_tail_float = max_train_tail_float if max_train_tail_float is not None else max_rho_p99_float
    if max_vtrace_rho_p99 is not None and guard_tail_float is not None and guard_tail_float > float(max_vtrace_rho_p99):
        failures.append(
            {
                "code": "vtrace_train_rho_tail_exceeded"
                if max_train_tail_float is not None
                else "vtrace_rho_p99_exceeded",
                "observed": guard_tail_float,
                "raw_vtrace_rho_p99": max_rho_p99_float,
                "train_vtrace_rho_tail": max_train_tail_float,
                "threshold": float(max_vtrace_rho_p99),
            }
        )
    return {
        "kind": "league_guard_v1",
        "passed": not failures,
        "failures": failures,
        "required_anchors": list(required_anchor_tuple),
        "latest_anchor_scores": anchor_scores,
        "min_latest_anchor_score": min_latest_anchor_score,
        "max_latest_drop": max_latest_drop,
        "require_promotion_pass_after_attempts": require_promotion_pass_after_attempts,
        "max_consecutive_promotion_failures": max_consecutive_promotion_failures,
        "max_vtrace_rho_p99": max_vtrace_rho_p99,
        "vtrace_guard_tail_source": "train" if max_train_tail_float is not None else "raw",
    }


__all__ = [
    "DEFAULT_LEAGUE_GUARD_ANCHORS",
    "_finite_float",
    "_latest_periodic_anchor_scores",
    "evaluate_league_guard",
]
