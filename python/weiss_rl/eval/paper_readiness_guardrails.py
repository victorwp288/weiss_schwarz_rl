"""Final-eval guardrail checks for paper-readiness reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scipy.stats import beta as beta_dist

from weiss_rl.eval.paper_readiness_fields import (
    as_int,
    as_optional_float,
    load_json_object,
    mapping,
)
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    LEGACY_NO_LEAGUE_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)


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


def build_baseline_check(
    payload: Mapping[str, Any],
    *,
    policy_ids: Sequence[str],
    focal_policy_id: str | None,
    baseline_policy_id: str,
    win_rate_threshold: float,
    posterior_min: float,
) -> dict[str, Any]:
    resolved_focal_policy_id = focal_policy_id
    focal_policy_source = "explicit_arg" if focal_policy_id is not None else None
    inferred_eligible_policy_ids: list[str] | None = None

    if resolved_focal_policy_id is None:
        inferred = infer_focal_policy_id(
            payload,
            policy_ids,
            baseline_policy_id=baseline_policy_id,
        )
        resolved_focal_policy_id = cast(str | None, inferred["focal_policy_id"])
        focal_policy_source = cast(str | None, inferred["source"])
        inferred_eligible_policy_ids = cast(list[str] | None, inferred.get("eligible_non_baseline_policy_ids"))

    result: dict[str, Any] = {
        "passed": False,
        "baseline_policy_id": baseline_policy_id,
        "focal_policy_id": resolved_focal_policy_id,
        "focal_policy_source": focal_policy_source,
        "win_rate_threshold": win_rate_threshold,
        "posterior_probability_threshold": posterior_min,
    }
    if inferred_eligible_policy_ids is not None:
        result["eligible_non_baseline_policy_ids"] = inferred_eligible_policy_ids

    if baseline_policy_id not in policy_ids:
        result["reason"] = "baseline_policy_missing_from_final_eval"
        return result
    if resolved_focal_policy_id is None:
        if inferred_eligible_policy_ids:
            result["reason"] = "ambiguous_non_baseline_focal_policy"
            result["message"] = (
                "multiple eligible non-baseline policies found; "
                "pass --focal-policy-id to choose the focal policy explicitly"
            )
        else:
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
    posterior_sample_values = posterior_samples(payload, focal_index=focal_index, opponent_index=baseline_index)
    has_payoff_samples = bool(matrix_cell(payload, field="has_payoff_samples", row=focal_index, column=baseline_index))
    mean = as_optional_float(matrix_cell(payload, field="mean", row=focal_index, column=baseline_index))
    ci_low = as_optional_float(matrix_cell(payload, field="ci_low", row=focal_index, column=baseline_index))
    ci_high = as_optional_float(matrix_cell(payload, field="ci_high", row=focal_index, column=baseline_index))
    paired_seed_count = as_int(
        matrix_cell(payload, field="paired_seed_count", row=focal_index, column=baseline_index),
        context="paired_seed_count",
    )
    stop_reason = str(matrix_cell(payload, field="stop_reason", row=focal_index, column=baseline_index))
    prob_gt_threshold = (
        sum(1 for sample in posterior_sample_values if sample > win_rate_threshold) / len(posterior_sample_values)
        if posterior_sample_values
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
            "sample_count": len(posterior_sample_values),
            "prob_gt_threshold": prob_gt_threshold,
        }
    )

    if not has_payoff_samples or mean is None or prob_gt_threshold is None:
        result["reason"] = "baseline_matchup_has_no_payoff_samples"
        return result

    result["passed"] = prob_gt_threshold >= posterior_min
    return result


def infer_focal_policy_id(
    payload: Mapping[str, Any],
    policy_ids: Sequence[str],
    *,
    baseline_policy_id: str,
) -> dict[str, Any]:
    metadata_policy_id = metadata_focal_policy_id(payload)
    if metadata_policy_id is not None:
        return {
            "focal_policy_id": metadata_policy_id,
            "source": "metadata",
        }

    baseline_ids = {
        RANDOM_LEGAL_POLICY_ID,
        NO_LEAGUE_POLICY_ID,
        LEGACY_NO_LEAGUE_POLICY_ID,
        HEURISTIC_PUBLIC_POLICY_ID,
        HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
        HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
        baseline_policy_id,
    }
    eligible_policy_ids = [policy_id for policy_id in policy_ids if policy_id not in baseline_ids]
    if len(eligible_policy_ids) == 1:
        return {
            "focal_policy_id": eligible_policy_ids[0],
            "source": "sole_eligible_non_baseline",
        }
    return {
        "focal_policy_id": None,
        "source": None,
        "eligible_non_baseline_policy_ids": eligible_policy_ids,
    }


def policy_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_policy_ids = payload.get("policy_ids")
    if not isinstance(raw_policy_ids, list) or any(not isinstance(item, str) for item in raw_policy_ids):
        raise ValueError("final_eval summary must include string policy_ids")
    return list(raw_policy_ids)


def matchups(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include matchups")
    matchup_payloads: list[Mapping[str, Any]] = []
    for index, matchup in enumerate(raw_matchups):
        matchup_payloads.append(mapping(matchup, context=f"matchups[{index}]"))
    return matchup_payloads


def canonical_unordered_matchups(
    matchups: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected: dict[tuple[int, int], tuple[int, Mapping[str, Any]]] = {}
    for index, matchup in enumerate(matchups):
        focal_index = matchup_policy_index(
            matchup,
            index_field="focal_policy_index",
            policy_field="focal_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        opponent_index = matchup_policy_index(
            matchup,
            index_field="opponent_policy_index",
            policy_field="opponent_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
        rank = 0 if focal_index <= opponent_index else 1
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, matchup)
    return [selected[key][1] for key in sorted(selected)]


def load_matchup_diagnostics(
    *,
    final_eval_dir: Path,
    matchups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for matchup in matchups:
        diagnostics_path = final_eval_dir / str(matchup["diagnostics_path"])
        diagnostics = load_json_object(diagnostics_path)
        seat_results = mapping(diagnostics.get("seat_results"), context=f"{diagnostics_path}:seat_results")
        seat0_wins = as_int(seat_results.get("seat0_wins"), context=f"{diagnostics_path}:seat0_wins")
        seat1_wins = as_int(seat_results.get("seat1_wins"), context=f"{diagnostics_path}:seat1_wins")
        draws = as_int(seat_results.get("draws"), context=f"{diagnostics_path}:draws")
        truncations = as_int(seat_results.get("truncations"), context=f"{diagnostics_path}:truncations")
        engine_errors = as_int(seat_results.get("engine_errors"), context=f"{diagnostics_path}:engine_errors")
        decisive_games = seat0_wins + seat1_wins
        payloads.append(
            {
                "focal_policy_id": str(matchup["focal_policy_id"]),
                "opponent_policy_id": str(matchup["opponent_policy_id"]),
                "diagnostics_path": str(matchup["diagnostics_path"]),
                "seat0_wins": seat0_wins,
                "seat1_wins": seat1_wins,
                "draws": draws,
                "truncations": truncations,
                "engine_errors": engine_errors,
                "decisive_games": decisive_games,
                "total_games": decisive_games + draws + truncations,
            }
        )
    return payloads


def matchup_policy_index(
    matchup: Mapping[str, Any],
    *,
    index_field: str,
    policy_field: str,
    policy_ids: Sequence[str],
    context: str,
) -> int:
    raw_index = matchup.get(index_field)
    if raw_index is not None:
        index = as_int(raw_index, context=f"{context}.{index_field}")
        if index < 0 or index >= len(policy_ids):
            raise ValueError(
                f"{context}.{index_field}={index} is out of range for policy_ids with length {len(policy_ids)}"
            )
        return index
    policy_id = matchup.get(policy_field)
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"{context}.{policy_field} must be a non-empty string")
    try:
        return policy_ids.index(policy_id)
    except ValueError as exc:
        raise ValueError(f"{context}.{policy_field}={policy_id!r} is missing from policy_ids") from exc


def metadata_focal_policy_id(payload: Mapping[str, Any]) -> str | None:
    metadata = mapping(payload.get("metadata", {}), context="metadata")
    for path in (
        ("focal_policy_id",),
        ("recommended_focal_policy_id",),
        ("focal_policy", "policy_id"),
        ("selection", "focal_policy_id"),
    ):
        value = nested_optional_string(metadata, path=path)
        if value is not None:
            return value
    return None


def nested_optional_string(payload: Mapping[str, Any], *, path: Sequence[str]) -> str | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, str):
        normalized = current.strip()
        return normalized or None
    return None


def matrix_cell(payload: Mapping[str, Any], *, field: str, row: int, column: int) -> Any:
    payload_matrix = matrix(payload, field=field)
    try:
        matrix_row = payload_matrix[row]
        if not isinstance(matrix_row, list):
            raise TypeError
        return matrix_row[column]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"matrix {field!r} is missing cell [{row}][{column}]") from exc


def matrix(payload: Mapping[str, Any], *, field: str) -> list[Any]:
    matrices = mapping(payload.get("matrices"), context="matrices")
    matrix_payload = mapping(matrices.get(field), context=f"matrices.{field}")
    values = matrix_payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"matrices.{field}.values must be a list")
    return values


def posterior_samples(payload: Mapping[str, Any], *, focal_index: int, opponent_index: int) -> list[float]:
    posterior_payload = mapping(payload.get("posterior_samples"), context="posterior_samples")
    values = posterior_payload.get("values")
    if not isinstance(values, list):
        raise ValueError("posterior_samples.values must be a list")
    try:
        row = values[focal_index]
        if not isinstance(row, list):
            raise TypeError
        samples = row[opponent_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"posterior_samples.values is missing cell [{focal_index}][{opponent_index}]") from exc
    if not isinstance(samples, list):
        raise ValueError("posterior sample cell must be a list")
    return [float(sample) for sample in samples]
