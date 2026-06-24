"""Final-eval guardrail checks for paper-readiness reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scipy.stats import beta as beta_dist

from weiss_rl.eval.readiness import final_eval_summary as _final_eval
from weiss_rl.eval.readiness.baseline_guardrail import build_baseline_check, infer_focal_policy_id
from weiss_rl.eval.readiness.fields import (
    as_int,
    load_json_object,
)

canonical_unordered_matchups = _final_eval.canonical_unordered_matchups
load_matchup_diagnostics = _final_eval.load_matchup_diagnostics
matchup_policy_index = _final_eval.matchup_policy_index
matchups = _final_eval.matchups
matrix_cell = _final_eval.matrix_cell
nested_optional_string = _final_eval.nested_optional_string
policy_ids = _final_eval.policy_ids
posterior_samples = _final_eval.posterior_samples


def build_final_eval_guardrail_summary(
    *,
    final_eval_dir: Path,
    focal_policy_id: str | None,
    baseline_policy_id: str,
    max_truncation_rate: float,
    seat_bias_max_abs_delta: float,
    seat_bias_posterior_min: float,
    baseline_win_rate_threshold: float,
    baseline_posterior_min: float,
) -> dict[str, Any]:
    summary_path = final_eval_dir / "summary.json"
    payload = load_json_object(summary_path)
    ids = policy_ids(payload)
    matchup_payloads = matchups(payload)
    canonical_matchups = canonical_unordered_matchups(matchup_payloads, policy_ids=ids)
    matchup_diagnostics = load_matchup_diagnostics(final_eval_dir=final_eval_dir, matchups=canonical_matchups)

    truncation = build_truncation_check(
        matchup_diagnostics,
        max_truncation_rate=max_truncation_rate,
    )
    seat_bias = build_seat_bias_check(
        matchup_diagnostics=matchup_diagnostics,
        max_abs_delta=seat_bias_max_abs_delta,
        posterior_min=seat_bias_posterior_min,
    )
    baseline = build_baseline_check(
        payload,
        policy_ids=ids,
        focal_policy_id=focal_policy_id,
        baseline_policy_id=baseline_policy_id,
        win_rate_threshold=baseline_win_rate_threshold,
        posterior_min=baseline_posterior_min,
    )

    checks = {
        "truncation_rate": truncation,
        "seat_bias_alarm": seat_bias,
        "baseline_win_rate_vs_b0": baseline,
    }
    alarms = [name for name, check in checks.items() if not bool(check["passed"])]
    metadata = cast(Mapping[str, Any], payload.get("metadata", {}))

    return {
        "passed": not alarms,
        "alarms": alarms,
        "final_eval": {
            "dir": final_eval_dir.as_posix(),
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(ids),
            "selection": dict(cast(Mapping[str, Any], metadata.get("selection", {}))),
        },
        "checks": checks,
    }


def build_truncation_check(
    matchup_diagnostics: Sequence[Mapping[str, Any]],
    *,
    max_truncation_rate: float,
) -> dict[str, Any]:
    total_games = sum(as_int(matchup["total_games"], context="total_games") for matchup in matchup_diagnostics)
    truncated_games = sum(as_int(matchup["truncations"], context="truncations") for matchup in matchup_diagnostics)
    rate = (truncated_games / total_games) if total_games else None
    passed = total_games > 0 and rate is not None and rate <= max_truncation_rate
    result: dict[str, Any] = {
        "passed": passed,
        "truncated_games": truncated_games,
        "total_games": total_games,
        "rate": rate,
        "max_allowed_rate": max_truncation_rate,
    }
    if total_games == 0:
        result["reason"] = "final_eval_summary_contains_no_games"
    return result


def build_seat_bias_check(
    *,
    matchup_diagnostics: Sequence[Mapping[str, Any]],
    max_abs_delta: float,
    posterior_min: float,
) -> dict[str, Any]:
    per_matchup: list[dict[str, Any]] = []
    seat0_wins = 0
    seat1_wins = 0
    draws = 0
    truncations = 0
    engine_errors = 0

    for matchup in matchup_diagnostics:
        matchup_seat0_wins = as_int(matchup["seat0_wins"], context="seat0_wins")
        matchup_seat1_wins = as_int(matchup["seat1_wins"], context="seat1_wins")
        matchup_draws = as_int(matchup["draws"], context="draws")
        matchup_truncations = as_int(matchup["truncations"], context="truncations")
        matchup_engine_errors = as_int(matchup["engine_errors"], context="engine_errors")
        decisive_games = as_int(matchup["decisive_games"], context="decisive_games")

        seat0_wins += matchup_seat0_wins
        seat1_wins += matchup_seat1_wins
        draws += matchup_draws
        truncations += matchup_truncations
        engine_errors += matchup_engine_errors

        per_matchup.append(
            {
                "focal_policy_id": str(matchup["focal_policy_id"]),
                "opponent_policy_id": str(matchup["opponent_policy_id"]),
                "diagnostics_path": str(matchup["diagnostics_path"]),
                "seat0_wins": matchup_seat0_wins,
                "seat1_wins": matchup_seat1_wins,
                "decisive_games": decisive_games,
                "seat0_win_rate": (matchup_seat0_wins / decisive_games) if decisive_games else None,
                "seat1_win_rate": (matchup_seat1_wins / decisive_games) if decisive_games else None,
                "draws": matchup_draws,
                "truncations": matchup_truncations,
                "engine_errors": matchup_engine_errors,
            }
        )

    decisive_games = seat0_wins + seat1_wins
    result: dict[str, Any] = {
        "passed": False,
        "alarm": None,
        "observed": {
            "seat0_wins": seat0_wins,
            "seat1_wins": seat1_wins,
            "draws": draws,
            "truncations": truncations,
            "engine_errors": engine_errors,
            "decisive_games": decisive_games,
            "total_games": decisive_games + draws + truncations,
        },
        "thresholds": {
            "max_abs_delta_from_half": max_abs_delta,
            "posterior_probability": posterior_min,
        },
        "per_matchup": per_matchup,
    }
    if decisive_games == 0:
        result["reason"] = "seat_bias_requires_at_least_one_decisive_game"
        return result

    alpha = seat0_wins + 0.5
    beta_param = seat1_wins + 0.5
    ci_low, ci_high = beta_dist.ppf((0.025, 0.975), alpha, beta_param)
    prob_gt_upper = 1.0 - float(beta_dist.cdf(0.5 + max_abs_delta, alpha, beta_param))
    prob_lt_lower = float(beta_dist.cdf(0.5 - max_abs_delta, alpha, beta_param))
    alarm = prob_gt_upper > posterior_min or prob_lt_lower > posterior_min

    result["passed"] = not alarm
    result["alarm"] = alarm
    result["posterior"] = {
        "mean": float(alpha / (alpha + beta_param)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_gt_half_plus_delta": prob_gt_upper,
        "prob_lt_half_minus_delta": prob_lt_lower,
    }
    return result


__all__ = [
    "build_baseline_check",
    "build_final_eval_guardrail_summary",
    "build_seat_bias_check",
    "build_truncation_check",
    "infer_focal_policy_id",
    "matrix_cell",
    "posterior_samples",
]
