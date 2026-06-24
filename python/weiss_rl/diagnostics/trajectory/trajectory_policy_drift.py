"""State-matched policy drift summaries for replay trajectory checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import family_counts as _family_counts
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import family_name as _family_name
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import family_transitions as _family_transitions
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import finite_mean as _finite_mean
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import finite_values as _finite_values
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import flat_float as _flat_float
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import flat_int as _flat_int
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import flat_mask as _flat_mask
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import flat_optional as _flat_optional
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    largest_probability_drops as _largest_probability_drops,
)
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import margin_summary as _margin_summary
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    masked_same_family_rate as _masked_same_family_rate,
)
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import optional_difference as _optional_difference
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import percentiles as _percentiles
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    small_abs_delta_count as _small_abs_delta_count,
)
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    target_family_delta_summaries as _target_family_delta_summaries,
)
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    target_family_summaries as _target_family_summaries,
)
from weiss_rl.diagnostics.trajectory.trajectory_policy_drift_stats import (
    top_action_change_examples as _top_action_change_examples,
)


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


__all__ = [
    "_family_counts",
    "_family_name",
    "_family_transitions",
    "_finite_mean",
    "_finite_values",
    "_flat_float",
    "_flat_int",
    "_flat_mask",
    "_flat_optional",
    "_largest_probability_drops",
    "_margin_summary",
    "_masked_same_family_rate",
    "_optional_difference",
    "_percentiles",
    "_small_abs_delta_count",
    "_target_family_delta_summaries",
    "_target_family_summaries",
    "_top_action_change_examples",
    "summarize_policy_drift",
    "summarize_policy_drift_by_group",
    "summarize_policy_scores",
]
