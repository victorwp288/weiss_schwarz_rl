"""Summary aggregation helpers for replay inspection reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

_TRAJECTORY_NUMERIC_FIELDS = (
    "raw_legal_action_count",
    "self_level_count",
    "self_clock_count",
    "self_deck_count",
    "self_hand_count",
    "self_stock_count",
    "self_waiting_room_count",
    "self_memory_count",
    "self_climax_count",
    "self_stage_occupied_count",
    "opponent_level_count",
    "opponent_clock_count",
    "opponent_deck_count",
    "opponent_hand_count",
    "opponent_stock_count",
    "opponent_waiting_room_count",
    "opponent_memory_count",
    "opponent_climax_count",
    "opponent_stage_occupied_count",
)

TRACKED_LEGAL_FAMILIES = (
    "clock_from_hand",
    "main_play_character",
    "main_move",
    "climax_play",
    "attack",
    "pass",
)


def summarize_step_diffs(
    step_diffs: Sequence[dict[str, Any]],
    *,
    top_k: int,
    include_actor_summaries: bool = True,
) -> dict[str, Any]:
    if not step_diffs:
        return {
            "compared_steps": 0,
            "top_k": int(top_k),
            "max_total_variation": 0.0,
            "mean_total_variation": 0.0,
            "median_total_variation": 0.0,
            "max_abs_probability_delta": 0.0,
            "policy_a_matches_policy_b_top_action_rate": 0.0,
            "policy_a_matches_policy_b_top_action_family_rate": 0.0,
            "policy_a_mean_probability_on_policy_b_top_action": 0.0,
            "policy_a_mean_probability_on_policy_b_top_action_family": 0.0,
            "policy_a_median_rank_of_policy_b_top_action": 0.0,
            "policy_a_probability_on_policy_b_top_action_percentiles": percentile_summary([]),
            "policy_a_top_logit_margin_percentiles": percentile_summary([]),
            "policy_a_top_probability_margin_percentiles": percentile_summary([]),
            "policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles": percentile_summary([]),
            "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": percentile_summary([]),
            "raw_legal_action_count_percentiles": percentile_summary([]),
            "policy_a_legal_action_count_percentiles": percentile_summary([]),
            "policy_b_legal_action_count_percentiles": percentile_summary([]),
            "policy_a_legal_surface_filter_rate": 0.0,
            "policy_b_legal_surface_filter_rate": 0.0,
            "policy_a_mean_raw_minus_policy_a_legal_action_count": 0.0,
            "policy_b_mean_raw_minus_policy_b_legal_action_count": 0.0,
            "policy_b_top_action_illegal_for_policy_a_rate": 0.0,
            "policy_a_top_action_illegal_for_policy_b_rate": 0.0,
            "policy_b_top_family_summaries": [],
            "top_action_family_confusions": [],
            "policy_a_mean_family_probability_masses": [],
            "actor_summaries": [],
        }

    total_variation = np.asarray([float(item["total_variation"]) for item in step_diffs], dtype=np.float64)
    max_abs_probability_delta = np.asarray(
        [float(item["max_abs_probability_delta"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_matches_policy_b_top_action = np.asarray(
        [bool(item["policy_a_matches_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_matches_policy_b_top_action_family = np.asarray(
        [bool(item["policy_a_matches_policy_b_top_action_family"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_probability_on_policy_b_top_action = np.asarray(
        [float(item["policy_a_probability_on_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_probability_on_policy_b_top_action_family = np.asarray(
        [float(item["policy_a_probability_on_policy_b_top_action_family"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_rank_of_policy_b_top_action = np.asarray(
        [float(item["policy_a_rank_of_policy_b_top_action"]) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_top_logit_margin = finite_float_values(step_diffs, "policy_a_top_logit_margin")
    policy_a_top_probability_margin = finite_float_values(step_diffs, "policy_a_top_probability_margin")
    policy_a_gap_from_top_logit_to_policy_b_top_action = finite_float_values(
        step_diffs,
        "policy_a_gap_from_top_logit_to_policy_b_top_action",
    )
    policy_a_policy_b_top_action_same_family_logit_margin = finite_float_values(
        step_diffs,
        "policy_a_policy_b_top_action_same_family_logit_margin",
    )
    raw_legal_action_counts = finite_float_values(step_diffs, "raw_legal_action_count")
    policy_a_legal_action_counts = finite_float_values(step_diffs, "policy_a_legal_action_count")
    policy_b_legal_action_counts = finite_float_values(step_diffs, "policy_b_legal_action_count")
    policy_a_removed_counts = np.asarray(
        [float(item.get("policy_a_legal_surface_removed_action_count", 0.0)) for item in step_diffs],
        dtype=np.float64,
    )
    policy_b_removed_counts = np.asarray(
        [float(item.get("policy_b_legal_surface_removed_action_count", 0.0)) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_surface_filtered = np.asarray(
        [bool(item.get("policy_a_legal_surface_is_filtered", False)) for item in step_diffs],
        dtype=np.float64,
    )
    policy_b_surface_filtered = np.asarray(
        [bool(item.get("policy_b_legal_surface_is_filtered", False)) for item in step_diffs],
        dtype=np.float64,
    )
    policy_b_top_illegal_for_policy_a = np.asarray(
        [not bool(item.get("policy_b_top_action_legal_for_policy_a", True)) for item in step_diffs],
        dtype=np.float64,
    )
    policy_a_top_illegal_for_policy_b = np.asarray(
        [not bool(item.get("policy_a_top_action_legal_for_policy_b", True)) for item in step_diffs],
        dtype=np.float64,
    )
    confusion_counter: Counter[tuple[str, str]] = Counter()
    for item in step_diffs:
        policy_b_family = str(item["policy_b_top_action"].get("family", "unknown"))
        policy_a_family = str(item["policy_a_top_action"].get("family", "unknown"))
        confusion_counter[(policy_b_family, policy_a_family)] += 1
    policy_a_mean_family_masses = mean_family_probability_masses(
        item.get("policy_a_family_probability_masses", {}) for item in step_diffs
    )
    summary = {
        "compared_steps": len(step_diffs),
        "top_k": int(top_k),
        "max_total_variation": float(np.max(total_variation)),
        "mean_total_variation": float(np.mean(total_variation)),
        "median_total_variation": float(np.median(total_variation)),
        "max_abs_probability_delta": float(np.max(max_abs_probability_delta)),
        "policy_a_matches_policy_b_top_action_rate": float(np.mean(policy_a_matches_policy_b_top_action)),
        "policy_a_matches_policy_b_top_action_family_rate": float(np.mean(policy_a_matches_policy_b_top_action_family)),
        "policy_a_top_action_mismatch_count": int(len(step_diffs) - int(np.sum(policy_a_matches_policy_b_top_action))),
        "policy_a_top_action_family_mismatch_count": int(
            len(step_diffs) - int(np.sum(policy_a_matches_policy_b_top_action_family))
        ),
        "policy_a_mean_probability_on_policy_b_top_action": float(np.mean(policy_a_probability_on_policy_b_top_action)),
        "policy_a_mean_probability_on_policy_b_top_action_family": float(
            np.mean(policy_a_probability_on_policy_b_top_action_family)
        ),
        "policy_a_median_rank_of_policy_b_top_action": float(np.median(policy_a_rank_of_policy_b_top_action)),
        "policy_a_probability_on_policy_b_top_action_percentiles": percentile_summary(
            policy_a_probability_on_policy_b_top_action.tolist()
        ),
        "policy_a_top_logit_margin_percentiles": percentile_summary(policy_a_top_logit_margin),
        "policy_a_top_probability_margin_percentiles": percentile_summary(policy_a_top_probability_margin),
        "policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles": percentile_summary(
            policy_a_gap_from_top_logit_to_policy_b_top_action
        ),
        "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": percentile_summary(
            policy_a_policy_b_top_action_same_family_logit_margin
        ),
        "raw_legal_action_count_percentiles": percentile_summary(raw_legal_action_counts),
        "policy_a_legal_action_count_percentiles": percentile_summary(policy_a_legal_action_counts),
        "policy_b_legal_action_count_percentiles": percentile_summary(policy_b_legal_action_counts),
        "policy_a_legal_surface_filter_rate": float(np.mean(policy_a_surface_filtered)),
        "policy_b_legal_surface_filter_rate": float(np.mean(policy_b_surface_filtered)),
        "policy_a_mean_raw_minus_policy_a_legal_action_count": float(np.mean(policy_a_removed_counts)),
        "policy_b_mean_raw_minus_policy_b_legal_action_count": float(np.mean(policy_b_removed_counts)),
        "policy_b_top_action_illegal_for_policy_a_rate": float(np.mean(policy_b_top_illegal_for_policy_a)),
        "policy_a_top_action_illegal_for_policy_b_rate": float(np.mean(policy_a_top_illegal_for_policy_b)),
        "policy_b_top_family_summaries": policy_b_top_family_summaries(step_diffs),
        "top_action_family_confusions": [
            {
                "policy_b_family": policy_b_family,
                "policy_a_family": policy_a_family,
                "count": int(count),
            }
            for (policy_b_family, policy_a_family), count in confusion_counter.most_common()
        ],
        "policy_a_mean_family_probability_masses": policy_a_mean_family_masses,
    }
    if include_actor_summaries:
        summary["actor_summaries"] = actor_summaries(step_diffs, top_k=top_k)
    return summary


def actor_summaries(step_diffs: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    by_actor: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in step_diffs:
        by_actor[int(item["actor"])].append(item)
    summaries: list[dict[str, Any]] = []
    for actor, actor_items in sorted(by_actor.items(), key=lambda item: item[0]):
        actor_summary = summarize_step_diffs(
            actor_items,
            top_k=top_k,
            include_actor_summaries=False,
        )
        summaries.append(
            {
                "actor": int(actor),
                "compared_steps": int(actor_summary["compared_steps"]),
                "mean_total_variation": float(actor_summary["mean_total_variation"]),
                "policy_a_matches_policy_b_top_action_rate": float(
                    actor_summary["policy_a_matches_policy_b_top_action_rate"]
                ),
                "policy_a_matches_policy_b_top_action_family_rate": float(
                    actor_summary["policy_a_matches_policy_b_top_action_family_rate"]
                ),
                "policy_a_mean_probability_on_policy_b_top_action": float(
                    actor_summary["policy_a_mean_probability_on_policy_b_top_action"]
                ),
                "policy_a_mean_probability_on_policy_b_top_action_family": float(
                    actor_summary["policy_a_mean_probability_on_policy_b_top_action_family"]
                ),
                "policy_a_median_rank_of_policy_b_top_action": float(
                    actor_summary["policy_a_median_rank_of_policy_b_top_action"]
                ),
                "top_action_family_confusions": actor_summary["top_action_family_confusions"],
                "policy_b_top_family_summaries": actor_summary["policy_b_top_family_summaries"],
            }
        )
    return summaries


def summarize_trajectory_records(
    records: Sequence[dict[str, Any]],
    *,
    include_actor_summaries: bool = True,
) -> dict[str, Any]:
    if not records:
        return {
            "compared_steps": 0,
            "recorded_family_counts": [],
            "phase_counts": [],
            "decision_kind_counts": [],
            "legal_family_presence_rates": [],
            "numeric_summaries": {},
            "recorded_family_summaries": [],
            "actor_summaries": [],
        }

    recorded_family_counter: Counter[str] = Counter(
        str(item.get("recorded_action_family", "unknown")) for item in records
    )
    phase_counter = value_counter(records, "phase")
    decision_kind_counter = value_counter(records, "decision_kind")
    legal_presence_rates = [
        {
            "family": family,
            "rate": float(np.mean([bool(item.get(f"has_legal_{family}", False)) for item in records])),
        }
        for family in TRACKED_LEGAL_FAMILIES
    ]
    numeric_summaries = {
        field: percentile_summary(finite_float_values(records, field))
        for field in _TRAJECTORY_NUMERIC_FIELDS
        if any(field in item for item in records)
    }
    by_recorded_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_recorded_family[str(item.get("recorded_action_family", "unknown"))].append(item)
    recorded_family_summaries: list[dict[str, Any]] = []
    for family, family_records in sorted(by_recorded_family.items(), key=lambda entry: (-len(entry[1]), entry[0])):
        family_payload: dict[str, Any] = {
            "family": family,
            "count": len(family_records),
            "rate": float(len(family_records) / len(records)),
            "numeric_means": {
                field: mean_or_none(finite_float_values(family_records, field))
                for field in _TRAJECTORY_NUMERIC_FIELDS
                if any(field in item for item in family_records)
            },
        }
        recorded_family_summaries.append(family_payload)

    summary: dict[str, Any] = {
        "compared_steps": len(records),
        "recorded_family_counts": counter_items(recorded_family_counter, key_names=("family",)),
        "phase_counts": counter_items(phase_counter, key_names=("phase",)),
        "decision_kind_counts": counter_items(decision_kind_counter, key_names=("decision_kind",)),
        "legal_family_presence_rates": legal_presence_rates,
        "numeric_summaries": numeric_summaries,
        "recorded_family_summaries": recorded_family_summaries,
    }
    if include_actor_summaries:
        summary["actor_summaries"] = trajectory_actor_summaries(records)
    return summary


def trajectory_actor_summaries(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_actor: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_actor[int(item["actor"])].append(item)
    summaries: list[dict[str, Any]] = []
    for actor, actor_records in sorted(by_actor.items(), key=lambda entry: entry[0]):
        actor_summary = summarize_trajectory_records(actor_records, include_actor_summaries=False)
        summaries.append(
            {
                "actor": int(actor),
                "compared_steps": int(actor_summary["compared_steps"]),
                "recorded_family_counts": actor_summary["recorded_family_counts"],
                "phase_counts": actor_summary["phase_counts"],
                "decision_kind_counts": actor_summary["decision_kind_counts"],
                "legal_family_presence_rates": actor_summary["legal_family_presence_rates"],
                "numeric_summaries": actor_summary["numeric_summaries"],
                "recorded_family_summaries": actor_summary["recorded_family_summaries"],
            }
        )
    return summaries


def value_counter(records: Sequence[Mapping[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in records:
        if key in item:
            counter[str(item[key])] += 1
    return counter


def counter_items(counter: Counter[Any], *, key_names: tuple[str, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, count in sorted(
        counter.items(), key=lambda item: (-int(item[1]), tuple(str(part) for part in tuple_value(item[0])))
    ):
        payload: dict[str, Any] = {"count": int(count)}
        for key_name, part in zip(key_names, tuple_value(key), strict=False):
            payload[key_name] = part
        items.append(payload)
    return items


def tuple_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def mean_or_none(values: Sequence[float]) -> float | None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    return float(np.mean(np.asarray(finite_values, dtype=np.float64)))


def finite_float_values(items: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def policy_b_top_family_summaries(step_diffs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in step_diffs:
        family = str(item["policy_b_top_action"].get("family", "unknown"))
        by_family[family].append(item)

    summaries: list[dict[str, Any]] = []
    for family, family_items in sorted(by_family.items(), key=lambda item: (-len(item[1]), item[0])):
        action_matches = np.asarray(
            [bool(item["policy_a_matches_policy_b_top_action"]) for item in family_items],
            dtype=np.float64,
        )
        family_matches = np.asarray(
            [bool(item["policy_a_matches_policy_b_top_action_family"]) for item in family_items],
            dtype=np.float64,
        )
        probabilities = np.asarray(
            [float(item["policy_a_probability_on_policy_b_top_action"]) for item in family_items],
            dtype=np.float64,
        )
        family_probabilities = np.asarray(
            [float(item["policy_a_probability_on_policy_b_top_action_family"]) for item in family_items],
            dtype=np.float64,
        )
        policy_b_top_action_legal_for_policy_a = np.asarray(
            [bool(item.get("policy_b_top_action_legal_for_policy_a", True)) for item in family_items],
            dtype=np.float64,
        )
        policy_a_surface_filtered = np.asarray(
            [bool(item.get("policy_a_legal_surface_is_filtered", False)) for item in family_items],
            dtype=np.float64,
        )
        policy_a_removed_counts = np.asarray(
            [float(item.get("policy_a_legal_surface_removed_action_count", 0.0)) for item in family_items],
            dtype=np.float64,
        )
        same_family_margins = finite_float_values(
            family_items,
            "policy_a_policy_b_top_action_same_family_logit_margin",
        )
        summaries.append(
            {
                "family": family,
                "count": len(family_items),
                "policy_a_matches_policy_b_top_action_rate": float(np.mean(action_matches)),
                "policy_a_matches_policy_b_top_action_family_rate": float(np.mean(family_matches)),
                "policy_a_mean_probability_on_policy_b_top_action": float(np.mean(probabilities)),
                "policy_a_mean_probability_on_policy_b_top_action_family": float(np.mean(family_probabilities)),
                "policy_b_top_action_legal_for_policy_a_rate": float(np.mean(policy_b_top_action_legal_for_policy_a)),
                "policy_a_legal_surface_filter_rate": float(np.mean(policy_a_surface_filtered)),
                "policy_a_mean_raw_minus_policy_a_legal_action_count": float(np.mean(policy_a_removed_counts)),
                "policy_a_probability_on_policy_b_top_action_percentiles": percentile_summary(probabilities.tolist()),
                "policy_a_policy_b_top_action_same_family_logit_margin_percentiles": percentile_summary(
                    same_family_margins
                ),
            }
        )
    return summaries


def percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite_values = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite_values.size == 0:
        return {"count": 0, "mean": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "p10": float(np.percentile(finite_values, 10)),
        "p25": float(np.percentile(finite_values, 25)),
        "p50": float(np.percentile(finite_values, 50)),
        "p75": float(np.percentile(finite_values, 75)),
        "p90": float(np.percentile(finite_values, 90)),
    }


def mean_family_probability_masses(family_masses: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    count = 0
    for masses in family_masses:
        count += 1
        for family, mass in masses.items():
            family_name = str(family)
            totals[family_name] = totals.get(family_name, 0.0) + float(mass)
    if count == 0:
        return []
    return [
        {"family": family, "mean_probability": float(total / count)}
        for family, total in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]


def top_step_diffs(step_diffs: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if top_k == 0:
        return []
    ranked = sorted(
        step_diffs,
        key=lambda item: (
            float(item["total_variation"]),
            float(item["max_abs_probability_delta"]),
            -int(item["step_index"]),
        ),
        reverse=True,
    )
    return list(ranked[:top_k])


def canonical_float(value: Any) -> float:
    scalar = float(np.float32(value))
    return scalar if math.isfinite(scalar) else scalar


__all__ = [
    "TRACKED_LEGAL_FAMILIES",
    "canonical_float",
    "counter_items",
    "summarize_step_diffs",
    "summarize_trajectory_records",
    "top_step_diffs",
]
