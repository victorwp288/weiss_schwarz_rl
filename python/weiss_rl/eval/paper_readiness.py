"""Paper-readiness guardrails over final-eval artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, cast

from scipy.stats import beta as beta_dist

from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)

DEFAULT_BASELINE_POLICY_ID = RANDOM_LEGAL_POLICY_ID
DEFAULT_BASELINE_POSTERIOR_MIN = 0.95
DEFAULT_BASELINE_WIN_RATE_THRESHOLD = 0.55
DEFAULT_SEAT_BIAS_MAX_ABS_DELTA = 0.05
DEFAULT_SEAT_BIAS_POSTERIOR_MIN = 0.95
DEFAULT_TRUNCATION_MAX_RATE = 0.02

__all__ = [
    "DEFAULT_BASELINE_POLICY_ID",
    "DEFAULT_BASELINE_POSTERIOR_MIN",
    "DEFAULT_BASELINE_WIN_RATE_THRESHOLD",
    "DEFAULT_SEAT_BIAS_MAX_ABS_DELTA",
    "DEFAULT_SEAT_BIAS_POSTERIOR_MIN",
    "DEFAULT_TRUNCATION_MAX_RATE",
    "build_paper_readiness_summary",
    "write_paper_readiness_json",
]


def build_paper_readiness_summary(
    *,
    final_eval_dir: Path,
    focal_policy_id: str | None = None,
    baseline_policy_id: str = DEFAULT_BASELINE_POLICY_ID,
    max_truncation_rate: float = DEFAULT_TRUNCATION_MAX_RATE,
    seat_bias_max_abs_delta: float = DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
    seat_bias_posterior_min: float = DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
    baseline_win_rate_threshold: float = DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
    baseline_posterior_min: float = DEFAULT_BASELINE_POSTERIOR_MIN,
) -> dict[str, Any]:
    summary_path = final_eval_dir / "summary.json"
    payload = _load_json_object(summary_path)
    policy_ids = _policy_ids(payload)
    matchups = _matchups(payload)

    truncation = _build_truncation_check(
        payload,
        max_truncation_rate=max_truncation_rate,
    )
    seat_bias = _build_seat_bias_check(
        final_eval_dir=final_eval_dir,
        matchups=matchups,
        max_abs_delta=seat_bias_max_abs_delta,
        posterior_min=seat_bias_posterior_min,
    )
    baseline = _build_baseline_check(
        payload,
        policy_ids=policy_ids,
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
        "kind": "paper_readiness_summary_v1",
        "passed": not alarms,
        "alarms": alarms,
        "final_eval": {
            "dir": final_eval_dir.as_posix(),
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "selection": dict(cast(Mapping[str, Any], metadata.get("selection", {}))),
        },
        "checks": checks,
    }


def write_paper_readiness_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_truncation_check(payload: Mapping[str, Any], *, max_truncation_rate: float) -> dict[str, Any]:
    total_games = _sum_matrix_ints(payload, field="games")
    truncated_games = _sum_matrix_ints(payload, field="truncations")
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


def _build_seat_bias_check(
    *,
    final_eval_dir: Path,
    matchups: Sequence[Mapping[str, Any]],
    max_abs_delta: float,
    posterior_min: float,
) -> dict[str, Any]:
    per_matchup: list[dict[str, Any]] = []
    seat0_wins = 0
    seat1_wins = 0
    draws = 0
    truncations = 0
    engine_errors = 0

    for matchup in matchups:
        diagnostics_path = final_eval_dir / str(matchup["diagnostics_path"])
        diagnostics = _load_json_object(diagnostics_path)
        seat_results = _mapping(diagnostics.get("seat_results"), context=f"{diagnostics_path}:seat_results")
        matchup_seat0_wins = _as_int(seat_results.get("seat0_wins"), context=f"{diagnostics_path}:seat0_wins")
        matchup_seat1_wins = _as_int(seat_results.get("seat1_wins"), context=f"{diagnostics_path}:seat1_wins")
        matchup_draws = _as_int(seat_results.get("draws"), context=f"{diagnostics_path}:draws")
        matchup_truncations = _as_int(seat_results.get("truncations"), context=f"{diagnostics_path}:truncations")
        matchup_engine_errors = _as_int(
            seat_results.get("engine_errors"),
            context=f"{diagnostics_path}:engine_errors",
        )
        decisive_games = matchup_seat0_wins + matchup_seat1_wins

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


def _build_baseline_check(
    payload: Mapping[str, Any],
    *,
    policy_ids: Sequence[str],
    focal_policy_id: str | None,
    baseline_policy_id: str,
    win_rate_threshold: float,
    posterior_min: float,
) -> dict[str, Any]:
    resolved_focal_policy_id = focal_policy_id or _infer_focal_policy_id(
        policy_ids,
        baseline_policy_id=baseline_policy_id,
    )
    result: dict[str, Any] = {
        "passed": False,
        "baseline_policy_id": baseline_policy_id,
        "focal_policy_id": resolved_focal_policy_id,
        "win_rate_threshold": win_rate_threshold,
        "posterior_probability_threshold": posterior_min,
    }

    if baseline_policy_id not in policy_ids:
        result["reason"] = "baseline_policy_missing_from_final_eval"
        return result
    if resolved_focal_policy_id is None:
        result["reason"] = "could_not_infer_non_baseline_focal_policy"
        return result
    if resolved_focal_policy_id not in policy_ids:
        result["reason"] = "focal_policy_missing_from_final_eval"
        return result
    if resolved_focal_policy_id == baseline_policy_id:
        result["reason"] = "focal_policy_matches_baseline_policy"
        return result

    focal_index = policy_ids.index(resolved_focal_policy_id)
    baseline_index = policy_ids.index(baseline_policy_id)
    posterior_samples = _posterior_samples(payload, focal_index=focal_index, opponent_index=baseline_index)
    has_payoff_samples = bool(_matrix_cell(payload, field="has_payoff_samples", row=focal_index, column=baseline_index))
    mean = _as_optional_float(_matrix_cell(payload, field="mean", row=focal_index, column=baseline_index))
    ci_low = _as_optional_float(_matrix_cell(payload, field="ci_low", row=focal_index, column=baseline_index))
    ci_high = _as_optional_float(_matrix_cell(payload, field="ci_high", row=focal_index, column=baseline_index))
    paired_seed_count = _as_int(
        _matrix_cell(payload, field="paired_seed_count", row=focal_index, column=baseline_index),
        context="paired_seed_count",
    )
    stop_reason = str(_matrix_cell(payload, field="stop_reason", row=focal_index, column=baseline_index))
    prob_gt_threshold = (
        sum(1 for sample in posterior_samples if sample > win_rate_threshold) / len(posterior_samples)
        if posterior_samples
        else None
    )

    result.update(
        {
            "has_payoff_samples": has_payoff_samples,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "paired_seed_count": paired_seed_count,
            "stop_reason": stop_reason,
            "sample_count": len(posterior_samples),
            "prob_gt_threshold": prob_gt_threshold,
        }
    )

    if not has_payoff_samples or mean is None or prob_gt_threshold is None:
        result["reason"] = "baseline_matchup_has_no_payoff_samples"
        return result

    result["passed"] = prob_gt_threshold >= posterior_min
    return result


def _infer_focal_policy_id(policy_ids: Sequence[str], *, baseline_policy_id: str) -> str | None:
    baseline_ids = {
        RANDOM_LEGAL_POLICY_ID,
        NO_LEAGUE_POLICY_ID,
        HEURISTIC_PUBLIC_POLICY_ID,
        baseline_policy_id,
    }
    for policy_id in policy_ids:
        if policy_id not in baseline_ids:
            return policy_id
    for policy_id in policy_ids:
        if policy_id != baseline_policy_id:
            return policy_id
    return None


def _policy_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_policy_ids = payload.get("policy_ids")
    if not isinstance(raw_policy_ids, list) or any(not isinstance(item, str) for item in raw_policy_ids):
        raise ValueError("final_eval summary must include string policy_ids")
    return list(raw_policy_ids)


def _matchups(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include matchups")
    matchups: list[Mapping[str, Any]] = []
    for index, matchup in enumerate(raw_matchups):
        matchups.append(_mapping(matchup, context=f"matchups[{index}]"))
    return matchups


def _sum_matrix_ints(payload: Mapping[str, Any], *, field: str) -> int:
    matrix = _matrix(payload, field=field)
    total = 0
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ValueError(f"matrix {field!r} row {row_index} must be a list")
        for column_index, value in enumerate(row):
            total += _as_int(value, context=f"{field}[{row_index}][{column_index}]")
    return total


def _matrix_cell(payload: Mapping[str, Any], *, field: str, row: int, column: int) -> Any:
    matrix = _matrix(payload, field=field)
    try:
        matrix_row = matrix[row]
        if not isinstance(matrix_row, list):
            raise TypeError
        return matrix_row[column]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"matrix {field!r} is missing cell [{row}][{column}]") from exc


def _matrix(payload: Mapping[str, Any], *, field: str) -> list[Any]:
    matrices = _mapping(payload.get("matrices"), context="matrices")
    matrix_payload = _mapping(matrices.get(field), context=f"matrices.{field}")
    values = matrix_payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"matrices.{field}.values must be a list")
    return values


def _posterior_samples(payload: Mapping[str, Any], *, focal_index: int, opponent_index: int) -> list[float]:
    posterior_payload = _mapping(payload.get("posterior_samples"), context="posterior_samples")
    values = posterior_payload.get("values")
    if not isinstance(values, list):
        raise ValueError("posterior_samples.values must be a list")
    try:
        row = values[focal_index]
        if not isinstance(row, list):
            raise TypeError
        samples = row[opponent_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"posterior_samples.values is missing cell [{focal_index}][{opponent_index}]"
        ) from exc
    if not isinstance(samples, list):
        raise ValueError("posterior sample cell must be a list")
    return [float(sample) for sample in samples]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], payload)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _as_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric matrix cell or null")
    return float(value)
