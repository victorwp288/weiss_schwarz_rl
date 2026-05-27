"""Payoff and diagnostic matrix construction for final evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

MATRIX_FIELDS: tuple[str, ...] = (
    "mean",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "prob_gt_half",
    "prob_lt_half",
    "paired_seed_count",
    "observed_paired_seeds",
    "excluded_paired_seeds",
    "has_payoff_samples",
    "games",
    "wins",
    "losses",
    "draws",
    "truncations",
    "engine_errors",
    "stop_reason",
    "should_stop",
)


def build_matrix(
    *,
    policy_ids: Sequence[str],
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        [
            matrix_cell_value(
                canonical_results_by_key=canonical_results_by_key,
                focal_index=focal_index,
                opponent_index=opponent_index,
                field=field,
            )
            for opponent_index, _opponent_policy_id in enumerate(policy_ids)
        ]
        for focal_index, _focal_policy_id in enumerate(policy_ids)
    ]
    return {
        "policy_ids": list(policy_ids),
        "values": values,
    }


def matrix_cell_value(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
    field: str,
) -> Any:
    result, reverse = canonical_result_for_cell(
        canonical_results_by_key=canonical_results_by_key,
        focal_index=focal_index,
        opponent_index=opponent_index,
    )
    payload = cast(Mapping[str, Any], result["summary"])
    return matrix_value(payload, field=field, reverse=reverse)


def matrix_value(payload: Mapping[str, Any], *, field: str, reverse: bool = False) -> Any:
    summary = cast(Mapping[str, Any], payload["summary"])
    uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
    if reverse:
        return reverse_matrix_value(payload=payload, summary=summary, uncertainty=uncertainty, field=field)
    if field in uncertainty:
        return uncertainty[field]
    if field in summary:
        return summary[field]
    if field == "paired_seed_count":
        return uncertainty["paired_seed_count"]
    return payload[field]


def reverse_matrix_value(
    *,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    field: str,
) -> Any:
    if field == "mean":
        return invert_optional_float(uncertainty["mean"])
    if field == "ci_low":
        return invert_optional_float(uncertainty["ci_high"])
    if field == "ci_high":
        return invert_optional_float(uncertainty["ci_low"])
    if field == "prob_gt_half":
        return uncertainty["prob_lt_half"]
    if field == "prob_lt_half":
        return uncertainty["prob_gt_half"]
    if field == "wins":
        return summary["losses"]
    if field == "losses":
        return summary["wins"]
    if field in uncertainty:
        return uncertainty[field]
    if field in summary:
        return summary[field]
    if field == "paired_seed_count":
        return uncertainty["paired_seed_count"]
    return payload[field]


def invert_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return 1.0 - float(value)


def canonical_result_for_cell(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
) -> tuple[dict[str, Any], bool]:
    canonical_key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
    return canonical_results_by_key[canonical_key], focal_index > opponent_index


def posterior_samples_cell(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
) -> list[float]:
    result, reverse = canonical_result_for_cell(
        canonical_results_by_key=canonical_results_by_key,
        focal_index=focal_index,
        opponent_index=opponent_index,
    )
    samples = cast(Sequence[float], result["posterior_samples"])
    if not reverse:
        return [float(sample) for sample in samples]
    return [1.0 - float(sample) for sample in samples]


def covered_matrix_cells(*, focal_index: int, opponent_index: int) -> list[dict[str, int]]:
    cells = [{"focal_policy_index": focal_index, "opponent_policy_index": opponent_index}]
    if focal_index != opponent_index:
        cells.append({"focal_policy_index": opponent_index, "opponent_policy_index": focal_index})
    return cells
