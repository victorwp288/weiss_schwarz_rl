"""Numeric, family, and example payload helpers for trajectory policy drift."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def flat_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.bool_).reshape(-1)


def flat_int(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def flat_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def flat_optional(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return flat_float(values)[mask]


def optional_difference(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    if left is None or right is None:
        return None
    return left - right


def finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def finite_mean(values: Sequence[float] | np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def percentiles(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    finite = finite_values(values)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
    }


def family_name(family_id: int, *, family_names: Sequence[str]) -> str:
    value = int(family_id)
    if 0 <= value < len(family_names):
        return str(family_names[value])
    return f"unknown:{value}"


def family_counts(families: np.ndarray, *, family_names: Sequence[str]) -> list[dict[str, Any]]:
    counts = Counter(int(value) for value in families.tolist())
    return [
        {"family": family_name(family_id, family_names=family_names), "count": int(count)}
        for family_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def target_family_summaries(
    *,
    target_family: np.ndarray,
    action_match: np.ndarray,
    family_match: np.ndarray,
    target_probability: np.ndarray,
    family_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id in sorted(set(int(value) for value in target_family.tolist())):
        selected = target_family == int(family_id)
        rows.append(
            {
                "family": family_name(family_id, family_names=family_names),
                "count": int(np.count_nonzero(selected)),
                "top_action_matches_target_rate": float(np.mean(action_match[selected])),
                "top_family_matches_target_rate": float(np.mean(family_match[selected])),
                "mean_probability_on_target_action": finite_mean(target_probability[selected]),
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["family"])))
    return rows


def target_family_delta_summaries(
    *,
    target_family: np.ndarray,
    ref_matches_target: np.ndarray,
    cand_matches_target: np.ndarray,
    ref_prob: np.ndarray,
    cand_prob: np.ndarray,
    family_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id in sorted(set(int(value) for value in target_family.tolist())):
        selected = target_family == int(family_id)
        ref_rate = float(np.mean(ref_matches_target[selected]))
        cand_rate = float(np.mean(cand_matches_target[selected]))
        ref_mean = finite_mean(ref_prob[selected])
        cand_mean = finite_mean(cand_prob[selected])
        rows.append(
            {
                "family": family_name(family_id, family_names=family_names),
                "count": int(np.count_nonzero(selected)),
                "reference_top_action_matches_target_rate": ref_rate,
                "candidate_top_action_matches_target_rate": cand_rate,
                "top_action_matches_target_rate_delta": cand_rate - ref_rate,
                "reference_mean_probability_on_target_action": ref_mean,
                "candidate_mean_probability_on_target_action": cand_mean,
                "mean_probability_on_target_action_delta": cand_mean - ref_mean,
            }
        )
    rows.sort(
        key=lambda item: (
            float(item["mean_probability_on_target_action_delta"]),
            float(item["top_action_matches_target_rate_delta"]),
            -int(item["count"]),
        )
    )
    return rows


def family_transitions(
    *,
    reference_family: np.ndarray,
    candidate_family: np.ndarray,
    family_names: Sequence[str],
) -> list[dict[str, Any]]:
    counts = Counter(
        (int(ref), int(cand))
        for ref, cand in zip(reference_family.tolist(), candidate_family.tolist(), strict=True)
        if int(ref) != int(cand)
    )
    return [
        {
            "reference_family": family_name(ref, family_names=family_names),
            "candidate_family": family_name(cand, family_names=family_names),
            "count": int(count),
        }
        for (ref, cand), count in counts.most_common()
    ]


def masked_same_family_rate(
    *,
    row_mask: np.ndarray,
    reference_family: np.ndarray,
    candidate_family: np.ndarray,
) -> float:
    selected = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
    if not bool(np.any(selected)):
        return 0.0
    return float(np.mean(reference_family[selected] == candidate_family[selected]))


def small_abs_delta_count(*, prob_delta: np.ndarray, row_mask: np.ndarray, threshold: float) -> int:
    selected = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
    if not bool(np.any(selected)):
        return 0
    values = np.asarray(prob_delta, dtype=np.float64).reshape(-1)[selected]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(finite) <= float(threshold)))


def margin_summary(margins: np.ndarray | None, row_mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
    values = np.asarray([], dtype=np.float64) if margins is None else np.asarray(margins, dtype=np.float64).reshape(-1)
    if values.shape != selected.shape:
        values = np.asarray([], dtype=np.float64)
        selected = np.asarray([], dtype=np.bool_)
    selected_values = values[selected] if selected.size else np.asarray([], dtype=np.float64)
    finite = selected_values[np.isfinite(selected_values)]
    count = int(finite.size)
    thresholds = (1e-6, 1e-5, 1e-4, 1e-3)
    near_tie = []
    for threshold in thresholds:
        threshold_count = int(np.count_nonzero(finite <= float(threshold))) if count else 0
        near_tie.append(
            {
                "threshold": float(threshold),
                "count": threshold_count,
                "rate": 0.0 if count == 0 else float(threshold_count / count),
            }
        )
    return {
        "count": count,
        "percentiles": percentiles(finite),
        "near_tie_thresholds": near_tie,
    }


def top_action_change_examples(
    *,
    selected: np.ndarray,
    prob_delta: np.ndarray,
    ref_prob: np.ndarray,
    cand_prob: np.ndarray,
    ref_top: np.ndarray,
    cand_top: np.ndarray,
    ref_family: np.ndarray,
    cand_family: np.ndarray,
    target: np.ndarray,
    target_family: np.ndarray,
    candidate_top_over_target_margin: np.ndarray | None,
    family_names: Sequence[str],
    coordinates: np.ndarray | None,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    selected = np.asarray(selected, dtype=np.bool_).reshape(-1)
    if not bool(np.any(selected)):
        return []
    finite_selected = selected & np.asarray([math.isfinite(float(value)) for value in prob_delta], dtype=np.bool_)
    selected_indices = np.nonzero(finite_selected)[0]
    if selected_indices.size == 0:
        return []
    ordered = selected_indices[np.argsort(np.abs(prob_delta[selected_indices]))[: int(max_examples)]]
    examples: list[dict[str, Any]] = []
    for row_index in ordered.tolist():
        coordinate = {}
        if coordinates is not None:
            raw_coordinate = coordinates[int(row_index)]
            if isinstance(raw_coordinate, Mapping):
                coordinate = dict(raw_coordinate)
        candidate_margin = None
        if candidate_top_over_target_margin is not None and int(row_index) < int(candidate_top_over_target_margin.size):
            raw_margin = float(candidate_top_over_target_margin[int(row_index)])
            candidate_margin = raw_margin if math.isfinite(raw_margin) else None
        examples.append(
            {
                **coordinate,
                "target_action": int(target[row_index]),
                "target_family": family_name(int(target_family[row_index]), family_names=family_names),
                "reference_top_action": int(ref_top[row_index]),
                "reference_top_family": family_name(int(ref_family[row_index]), family_names=family_names),
                "candidate_top_action": int(cand_top[row_index]),
                "candidate_top_family": family_name(int(cand_family[row_index]), family_names=family_names),
                "top_action_same_family": bool(int(ref_family[row_index]) == int(cand_family[row_index])),
                "reference_probability_on_target_action": float(ref_prob[row_index]),
                "candidate_probability_on_target_action": float(cand_prob[row_index]),
                "probability_delta": float(prob_delta[row_index]),
                "abs_probability_delta": float(abs(prob_delta[row_index])),
                "candidate_top_over_target_logp_margin": candidate_margin,
            }
        )
    return examples


def largest_probability_drops(
    *,
    prob_delta: np.ndarray,
    ref_prob: np.ndarray,
    cand_prob: np.ndarray,
    ref_top: np.ndarray,
    cand_top: np.ndarray,
    ref_family: np.ndarray,
    cand_family: np.ndarray,
    target: np.ndarray,
    target_family: np.ndarray,
    family_names: Sequence[str],
    coordinates: np.ndarray | None,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0 or prob_delta.size == 0:
        return []
    finite = np.asarray([math.isfinite(float(value)) for value in prob_delta], dtype=np.bool_)
    if not bool(np.any(finite)):
        return []
    finite_indices = np.nonzero(finite)[0]
    ordered = finite_indices[np.argsort(prob_delta[finite_indices])[: int(max_examples)]]
    examples: list[dict[str, Any]] = []
    for row_index in ordered.tolist():
        coordinate = {}
        if coordinates is not None:
            raw_coordinate = coordinates[int(row_index)]
            if isinstance(raw_coordinate, Mapping):
                coordinate = dict(raw_coordinate)
        examples.append(
            {
                **coordinate,
                "target_action": int(target[row_index]),
                "target_family": family_name(int(target_family[row_index]), family_names=family_names),
                "reference_top_action": int(ref_top[row_index]),
                "reference_top_family": family_name(int(ref_family[row_index]), family_names=family_names),
                "candidate_top_action": int(cand_top[row_index]),
                "candidate_top_family": family_name(int(cand_family[row_index]), family_names=family_names),
                "reference_probability_on_target_action": float(ref_prob[row_index]),
                "candidate_probability_on_target_action": float(cand_prob[row_index]),
                "probability_delta": float(prob_delta[row_index]),
            }
        )
    return examples
