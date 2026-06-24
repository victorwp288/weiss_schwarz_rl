"""Baseline win-rate guardrail for paper-readiness checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from weiss_rl.eval.policies.fixed_panel import (
    HEURISTIC_PUBLIC_AGGRO_POLICY_ID,
    HEURISTIC_PUBLIC_CONTROL_POLICY_ID,
    HEURISTIC_PUBLIC_POLICY_ID,
    LEGACY_NO_LEAGUE_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)
from weiss_rl.eval.readiness.fields import as_int, as_optional_float
from weiss_rl.eval.readiness.final_eval_summary import metadata_focal_policy_id, posterior_samples


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
    cell = _matrix_cell_reader(payload, focal_index=focal_index, baseline_index=baseline_index)
    has_payoff_samples = bool(cell("has_payoff_samples"))
    mean = as_optional_float(cell("mean"))
    ci_low = as_optional_float(cell("ci_low"))
    ci_high = as_optional_float(cell("ci_high"))
    paired_seed_count = as_int(cell("paired_seed_count"), context="paired_seed_count")
    stop_reason = str(cell("stop_reason"))
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


def _matrix_cell_reader(payload: Mapping[str, Any], *, focal_index: int, baseline_index: int):
    from weiss_rl.eval.readiness.final_eval_summary import matrix_cell

    return lambda field: matrix_cell(payload, field=field, row=focal_index, column=baseline_index)


__all__ = ["build_baseline_check", "infer_focal_policy_id"]
