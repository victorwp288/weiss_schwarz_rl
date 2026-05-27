"""State-matched policy drift summaries for replay trajectory checkpoints."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_policy_scores(
    *,
    label: str,
    top_actions: np.ndarray,
    target_actions: np.ndarray,
    target_probabilities: np.ndarray,
    target_log_probs: np.ndarray,
    top_families: np.ndarray,
    target_families: np.ndarray,
    row_mask: np.ndarray,
    family_names: Sequence[str],
    values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize one policy on the trainable replay rows."""

    mask = _flat_mask(row_mask)
    top = _flat_int(top_actions)[mask]
    target = _flat_int(target_actions)[mask]
    top_family = _flat_int(top_families)[mask]
    target_family = _flat_int(target_families)[mask]
    target_probability = _flat_float(target_probabilities)[mask]
    target_logp = _flat_float(target_log_probs)[mask]
    value_array = None if values is None else _flat_float(values)[mask]
    row_count = int(mask.sum())
    if row_count == 0:
        return {
            "label": str(label),
            "row_count": 0,
            "top_action_matches_target_rate": 0.0,
            "top_family_matches_target_rate": 0.0,
            "mean_probability_on_target_action": 0.0,
            "target_action_probability_percentiles": _percentiles([]),
            "target_action_logp_percentiles": _percentiles([]),
            "value_percentiles": _percentiles([]),
            "top_family_counts": [],
            "target_family_summaries": [],
        }

    action_match = top == target
    family_match = top_family == target_family
    return {
        "label": str(label),
        "row_count": row_count,
        "top_action_matches_target_rate": float(np.mean(action_match)),
        "top_family_matches_target_rate": float(np.mean(family_match)),
        "mean_probability_on_target_action": _finite_mean(target_probability),
        "target_action_probability_percentiles": _percentiles(target_probability),
        "target_action_logp_percentiles": _percentiles(target_logp),
        "value_percentiles": _percentiles([] if value_array is None else value_array),
        "top_family_counts": _family_counts(top_family, family_names=family_names),
        "target_family_summaries": _target_family_summaries(
            target_family=target_family,
            action_match=action_match,
            family_match=family_match,
            target_probability=target_probability,
            family_names=family_names,
        ),
    }


def summarize_policy_drift(
    *,
    reference_label: str,
    candidate_label: str,
    reference_top_actions: np.ndarray,
    candidate_top_actions: np.ndarray,
    reference_target_probabilities: np.ndarray,
    candidate_target_probabilities: np.ndarray,
    reference_top_families: np.ndarray,
    candidate_top_families: np.ndarray,
    target_actions: np.ndarray,
    target_families: np.ndarray,
    row_mask: np.ndarray,
    family_names: Sequence[str],
    reference_target_log_probs: np.ndarray | None = None,
    candidate_target_log_probs: np.ndarray | None = None,
    reference_top_log_probs: np.ndarray | None = None,
    candidate_top_log_probs: np.ndarray | None = None,
    reference_values: np.ndarray | None = None,
    candidate_values: np.ndarray | None = None,
    row_coordinates: Sequence[Mapping[str, Any]] | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare a candidate policy against a reference on identical replay rows."""

    mask = _flat_mask(row_mask)
    ref_top = _flat_int(reference_top_actions)[mask]
    cand_top = _flat_int(candidate_top_actions)[mask]
    ref_family = _flat_int(reference_top_families)[mask]
    cand_family = _flat_int(candidate_top_families)[mask]
    target = _flat_int(target_actions)[mask]
    target_family = _flat_int(target_families)[mask]
    ref_prob = _flat_float(reference_target_probabilities)[mask]
    cand_prob = _flat_float(candidate_target_probabilities)[mask]
    prob_delta = cand_prob - ref_prob
    cand_target_logp = _flat_optional(candidate_target_log_probs, mask)
    cand_top_logp = _flat_optional(candidate_top_log_probs, mask)
    cand_top_over_target_margin = _optional_difference(cand_top_logp, cand_target_logp)
    value_delta = None
    if reference_values is not None and candidate_values is not None:
        value_delta = _flat_float(candidate_values)[mask] - _flat_float(reference_values)[mask]
    row_count = int(mask.sum())
    if row_count == 0:
        return {
            "reference_label": str(reference_label),
            "candidate_label": str(candidate_label),
            "row_count": 0,
            "top_action_changed_rate": 0.0,
            "top_family_changed_rate": 0.0,
            "lost_target_top_action_rate": 0.0,
            "gained_target_top_action_rate": 0.0,
            "target_action_probability_delta_percentiles": _percentiles([]),
            "value_delta_percentiles": _percentiles([]),
            "target_family_delta_summaries": [],
            "top_family_transitions": [],
            "top_action_changed_same_family_rate": 0.0,
            "lost_target_top_action_same_family_rate": 0.0,
            "top_action_changed_probability_delta_percentiles": _percentiles([]),
            "lost_target_top_action_probability_delta_percentiles": _percentiles([]),
            "top_action_changed_abs_probability_delta_lte_1e-5_count": 0,
            "lost_target_top_action_abs_probability_delta_lte_1e-5_count": 0,
            "top_action_changed_candidate_top_over_target_margin": _margin_summary(None, np.asarray([], dtype=bool)),
            "lost_target_top_action_candidate_top_over_target_margin": _margin_summary(
                None, np.asarray([], dtype=bool)
            ),
            "top_action_change_examples": [],
            "lost_target_top_action_examples": [],
            "largest_target_probability_drops": [],
        }

    ref_matches_target = ref_top == target
    cand_matches_target = cand_top == target
    top_changed = ref_top != cand_top
    lost_target_top = ref_matches_target & ~cand_matches_target
    coordinates = None
    if row_coordinates is not None:
        coordinate_array = np.asarray(list(row_coordinates), dtype=object)
        coordinates = coordinate_array[mask]

    return {
        "reference_label": str(reference_label),
        "candidate_label": str(candidate_label),
        "row_count": row_count,
        "top_action_changed_rate": float(np.mean(top_changed)),
        "top_family_changed_rate": float(np.mean(ref_family != cand_family)),
        "lost_target_top_action_rate": float(np.mean(lost_target_top)),
        "gained_target_top_action_rate": float(np.mean(~ref_matches_target & cand_matches_target)),
        "mean_target_action_probability_delta": _finite_mean(prob_delta),
        "target_action_probability_delta_percentiles": _percentiles(prob_delta),
        "mean_value_delta": 0.0 if value_delta is None else _finite_mean(value_delta),
        "value_delta_percentiles": _percentiles([] if value_delta is None else value_delta),
        "top_action_changed_same_family_rate": _masked_same_family_rate(
            row_mask=top_changed,
            reference_family=ref_family,
            candidate_family=cand_family,
        ),
        "lost_target_top_action_same_family_rate": _masked_same_family_rate(
            row_mask=lost_target_top,
            reference_family=ref_family,
            candidate_family=cand_family,
        ),
        "top_action_changed_probability_delta_percentiles": _percentiles(prob_delta[top_changed]),
        "lost_target_top_action_probability_delta_percentiles": _percentiles(prob_delta[lost_target_top]),
        "top_action_changed_abs_probability_delta_lte_1e-5_count": _small_abs_delta_count(
            prob_delta=prob_delta,
            row_mask=top_changed,
            threshold=1e-5,
        ),
        "lost_target_top_action_abs_probability_delta_lte_1e-5_count": _small_abs_delta_count(
            prob_delta=prob_delta,
            row_mask=lost_target_top,
            threshold=1e-5,
        ),
        "top_action_changed_candidate_top_over_target_margin": _margin_summary(
            cand_top_over_target_margin,
            top_changed,
        ),
        "lost_target_top_action_candidate_top_over_target_margin": _margin_summary(
            cand_top_over_target_margin,
            lost_target_top,
        ),
        "target_family_delta_summaries": _target_family_delta_summaries(
            target_family=target_family,
            ref_matches_target=ref_matches_target,
            cand_matches_target=cand_matches_target,
            ref_prob=ref_prob,
            cand_prob=cand_prob,
            family_names=family_names,
        ),
        "top_family_transitions": _family_transitions(
            reference_family=ref_family,
            candidate_family=cand_family,
            family_names=family_names,
        ),
        "top_action_change_examples": _top_action_change_examples(
            selected=top_changed,
            prob_delta=prob_delta,
            ref_prob=ref_prob,
            cand_prob=cand_prob,
            ref_top=ref_top,
            cand_top=cand_top,
            ref_family=ref_family,
            cand_family=cand_family,
            target=target,
            target_family=target_family,
            candidate_top_over_target_margin=cand_top_over_target_margin,
            family_names=family_names,
            coordinates=coordinates,
            max_examples=max_examples,
        ),
        "lost_target_top_action_examples": _top_action_change_examples(
            selected=lost_target_top,
            prob_delta=prob_delta,
            ref_prob=ref_prob,
            cand_prob=cand_prob,
            ref_top=ref_top,
            cand_top=cand_top,
            ref_family=ref_family,
            cand_family=cand_family,
            target=target,
            target_family=target_family,
            candidate_top_over_target_margin=cand_top_over_target_margin,
            family_names=family_names,
            coordinates=coordinates,
            max_examples=max_examples,
        ),
        "largest_target_probability_drops": _largest_probability_drops(
            prob_delta=prob_delta,
            ref_prob=ref_prob,
            cand_prob=cand_prob,
            ref_top=ref_top,
            cand_top=cand_top,
            ref_family=ref_family,
            cand_family=cand_family,
            target=target,
            target_family=target_family,
            family_names=family_names,
            coordinates=coordinates,
            max_examples=max_examples,
        ),
    }


def summarize_policy_drift_by_group(
    *,
    group_name: str,
    group_labels: Sequence[Any] | np.ndarray,
    reference_label: str,
    candidate_label: str,
    reference_top_actions: np.ndarray,
    candidate_top_actions: np.ndarray,
    reference_target_probabilities: np.ndarray,
    candidate_target_probabilities: np.ndarray,
    reference_top_families: np.ndarray,
    candidate_top_families: np.ndarray,
    target_actions: np.ndarray,
    target_families: np.ndarray,
    row_mask: np.ndarray,
    family_names: Sequence[str],
    reference_target_log_probs: np.ndarray | None = None,
    candidate_target_log_probs: np.ndarray | None = None,
    reference_top_log_probs: np.ndarray | None = None,
    candidate_top_log_probs: np.ndarray | None = None,
    reference_values: np.ndarray | None = None,
    candidate_values: np.ndarray | None = None,
    row_coordinates: Sequence[Mapping[str, Any]] | None = None,
    max_examples: int = 20,
) -> list[dict[str, Any]]:
    """Compare policy drift separately for named row groups."""

    base_mask = _flat_mask(row_mask)
    labels = np.asarray(group_labels, dtype=object).reshape(-1)
    if labels.shape != base_mask.shape:
        raise ValueError("group_labels must have one entry per flattened row")
    unique_labels = sorted(
        {str(label) for label, keep in zip(labels.tolist(), base_mask.tolist(), strict=True) if keep and str(label)}
    )
    summaries: list[dict[str, Any]] = []
    for label in unique_labels:
        group_mask = base_mask & (labels == label)
        summary = summarize_policy_drift(
            reference_label=reference_label,
            candidate_label=candidate_label,
            reference_top_actions=reference_top_actions,
            candidate_top_actions=candidate_top_actions,
            reference_target_probabilities=reference_target_probabilities,
            candidate_target_probabilities=candidate_target_probabilities,
            reference_top_families=reference_top_families,
            candidate_top_families=candidate_top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=group_mask,
            family_names=family_names,
            reference_target_log_probs=reference_target_log_probs,
            candidate_target_log_probs=candidate_target_log_probs,
            reference_top_log_probs=reference_top_log_probs,
            candidate_top_log_probs=candidate_top_log_probs,
            reference_values=reference_values,
            candidate_values=candidate_values,
            row_coordinates=row_coordinates,
            max_examples=max_examples,
        )
        summary[str(group_name)] = label
        summaries.append(summary)
    summaries.sort(key=lambda item: str(item[str(group_name)]))
    return summaries


def _flat_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.bool_).reshape(-1)


def _flat_int(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _flat_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _flat_optional(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return _flat_float(values)[mask]


def _optional_difference(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    if left is None or right is None:
        return None
    return left - right


def _finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_mean(values: Sequence[float] | np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _percentiles(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    finite = _finite_values(values)
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


def _family_name(family_id: int, *, family_names: Sequence[str]) -> str:
    value = int(family_id)
    if 0 <= value < len(family_names):
        return str(family_names[value])
    return f"unknown:{value}"


def _family_counts(families: np.ndarray, *, family_names: Sequence[str]) -> list[dict[str, Any]]:
    counts = Counter(int(value) for value in families.tolist())
    return [
        {"family": _family_name(family_id, family_names=family_names), "count": int(count)}
        for family_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _target_family_summaries(
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
                "family": _family_name(family_id, family_names=family_names),
                "count": int(np.count_nonzero(selected)),
                "top_action_matches_target_rate": float(np.mean(action_match[selected])),
                "top_family_matches_target_rate": float(np.mean(family_match[selected])),
                "mean_probability_on_target_action": _finite_mean(target_probability[selected]),
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["family"])))
    return rows


def _target_family_delta_summaries(
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
        ref_mean = _finite_mean(ref_prob[selected])
        cand_mean = _finite_mean(cand_prob[selected])
        rows.append(
            {
                "family": _family_name(family_id, family_names=family_names),
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


def _family_transitions(
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
            "reference_family": _family_name(ref, family_names=family_names),
            "candidate_family": _family_name(cand, family_names=family_names),
            "count": int(count),
        }
        for (ref, cand), count in counts.most_common()
    ]


def _masked_same_family_rate(
    *,
    row_mask: np.ndarray,
    reference_family: np.ndarray,
    candidate_family: np.ndarray,
) -> float:
    selected = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
    if not bool(np.any(selected)):
        return 0.0
    return float(np.mean(reference_family[selected] == candidate_family[selected]))


def _small_abs_delta_count(*, prob_delta: np.ndarray, row_mask: np.ndarray, threshold: float) -> int:
    selected = np.asarray(row_mask, dtype=np.bool_).reshape(-1)
    if not bool(np.any(selected)):
        return 0
    values = np.asarray(prob_delta, dtype=np.float64).reshape(-1)[selected]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(finite) <= float(threshold)))


def _margin_summary(margins: np.ndarray | None, row_mask: np.ndarray) -> dict[str, Any]:
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
        "percentiles": _percentiles(finite),
        "near_tie_thresholds": near_tie,
    }


def _top_action_change_examples(
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
                "target_family": _family_name(int(target_family[row_index]), family_names=family_names),
                "reference_top_action": int(ref_top[row_index]),
                "reference_top_family": _family_name(int(ref_family[row_index]), family_names=family_names),
                "candidate_top_action": int(cand_top[row_index]),
                "candidate_top_family": _family_name(int(cand_family[row_index]), family_names=family_names),
                "top_action_same_family": bool(int(ref_family[row_index]) == int(cand_family[row_index])),
                "reference_probability_on_target_action": float(ref_prob[row_index]),
                "candidate_probability_on_target_action": float(cand_prob[row_index]),
                "probability_delta": float(prob_delta[row_index]),
                "abs_probability_delta": float(abs(prob_delta[row_index])),
                "candidate_top_over_target_logp_margin": candidate_margin,
            }
        )
    return examples


def _largest_probability_drops(
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
                "target_family": _family_name(int(target_family[row_index]), family_names=family_names),
                "reference_top_action": int(ref_top[row_index]),
                "reference_top_family": _family_name(int(ref_family[row_index]), family_names=family_names),
                "candidate_top_action": int(cand_top[row_index]),
                "candidate_top_family": _family_name(int(cand_family[row_index]), family_names=family_names),
                "reference_probability_on_target_action": float(ref_prob[row_index]),
                "candidate_probability_on_target_action": float(cand_prob[row_index]),
                "probability_delta": float(prob_delta[row_index]),
            }
        )
    return examples


__all__ = ["summarize_policy_drift", "summarize_policy_scores"]
