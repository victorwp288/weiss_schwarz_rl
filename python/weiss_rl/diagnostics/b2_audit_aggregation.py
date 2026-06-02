"""Aggregation helpers for the B2 disagreement audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

_PERCENTILE_SUMMARY_FIELDS = ("mean", "p10", "p25", "p50", "p75", "p90")
_BUNDLE_PERCENTILE_SUMMARY_KEYS = (
    "policy_a_probability_on_policy_b_top_action_percentiles",
    "policy_a_top_logit_margin_percentiles",
    "policy_a_top_probability_margin_percentiles",
    "policy_a_gap_from_top_logit_to_policy_b_top_action_percentiles",
    "policy_a_policy_b_top_action_same_family_logit_margin_percentiles",
    "raw_legal_action_count_percentiles",
    "policy_a_legal_action_count_percentiles",
    "policy_b_legal_action_count_percentiles",
)
_POLICY_B_FAMILY_WEIGHTED_FIELDS = (
    "policy_a_matches_policy_b_top_action_rate",
    "policy_a_matches_policy_b_top_action_family_rate",
    "policy_a_mean_probability_on_policy_b_top_action",
    "policy_a_mean_probability_on_policy_b_top_action_family",
    "policy_b_top_action_legal_for_policy_a_rate",
    "policy_a_legal_surface_filter_rate",
    "policy_a_mean_raw_minus_policy_a_legal_action_count",
)
_POLICY_B_FAMILY_PERCENTILE_SUMMARY_KEYS = (
    "policy_a_probability_on_policy_b_top_action_percentiles",
    "policy_a_policy_b_top_action_same_family_logit_margin_percentiles",
)


def aggregate_audit_summary(
    *,
    source: Any,
    policy_id: str,
    opponent_policy_id: str,
    episodes_jsonl: Path,
    run_dir: Path,
    output_run_dir: Path,
    episodes_path: Path,
    game_count: int,
    bundle_summaries: Sequence[dict[str, Any]],
    inspection_errors: Sequence[dict[str, Any]],
    stack_config_hash256: str | None = None,
    run_manifest_config_hash256: str | None = None,
    policy_id_mismatch_allowed: bool = False,
    requested_policy_id: str | None = None,
) -> dict[str, Any]:
    family_pair_counts: Counter[tuple[str, str]] = Counter()
    policy_a_family_counts: Counter[str] = Counter()
    policy_b_family_counts: Counter[str] = Counter()
    recorded_family_counts: Counter[str] = Counter()
    action_label_pair_counts: Counter[tuple[str, str]] = Counter()
    policy_a_action_label_counts: Counter[str] = Counter()
    policy_b_action_label_counts: Counter[str] = Counter()
    all_step_family_confusions: Counter[tuple[str, str]] = Counter()
    compared_steps = 0
    inspected_steps = 0
    max_total_variation = 0.0
    weighted_total_variation = 0.0
    weighted_top_action_match_rate = 0.0
    weighted_top_action_family_match_rate = 0.0
    weighted_probability_on_policy_b_top_action = 0.0
    weighted_probability_on_policy_b_top_action_family = 0.0
    weighted_median_rank_of_policy_b_top_action = 0.0
    weighted_policy_a_legal_surface_filter_rate = 0.0
    weighted_policy_b_legal_surface_filter_rate = 0.0
    weighted_policy_a_removed_action_count = 0.0
    weighted_policy_b_removed_action_count = 0.0
    weighted_policy_b_top_action_illegal_for_policy_a_rate = 0.0
    weighted_policy_a_top_action_illegal_for_policy_b_rate = 0.0
    weighted_probability_weight = 0
    weighted_family_probability_masses: defaultdict[str, float] = defaultdict(float)
    top_examples: list[dict[str, Any]] = []

    for bundle_summary in bundle_summaries:
        compared_steps += int(bundle_summary["compared_steps"])
        inspected_steps += int(bundle_summary["inspected_step_count"])
        summary = bundle_summary["summary"]
        summary_weight = int(summary.get("compared_steps", bundle_summary["compared_steps"]))
        max_total_variation = max(max_total_variation, float(summary.get("max_total_variation", 0.0)))
        weighted_total_variation += float(summary.get("mean_total_variation", 0.0)) * summary_weight
        if summary_weight > 0:
            weighted_probability_weight += summary_weight
            weighted_top_action_match_rate += (
                float(summary.get("policy_a_matches_policy_b_top_action_rate", 0.0)) * summary_weight
            )
            weighted_top_action_family_match_rate += (
                float(summary.get("policy_a_matches_policy_b_top_action_family_rate", 0.0)) * summary_weight
            )
            weighted_probability_on_policy_b_top_action += (
                float(summary.get("policy_a_mean_probability_on_policy_b_top_action", 0.0)) * summary_weight
            )
            weighted_probability_on_policy_b_top_action_family += (
                float(summary.get("policy_a_mean_probability_on_policy_b_top_action_family", 0.0)) * summary_weight
            )
            weighted_median_rank_of_policy_b_top_action += (
                float(summary.get("policy_a_median_rank_of_policy_b_top_action", 0.0)) * summary_weight
            )
            weighted_policy_a_legal_surface_filter_rate += (
                float(summary.get("policy_a_legal_surface_filter_rate", 0.0)) * summary_weight
            )
            weighted_policy_b_legal_surface_filter_rate += (
                float(summary.get("policy_b_legal_surface_filter_rate", 0.0)) * summary_weight
            )
            weighted_policy_a_removed_action_count += (
                float(summary.get("policy_a_mean_raw_minus_policy_a_legal_action_count", 0.0)) * summary_weight
            )
            weighted_policy_b_removed_action_count += (
                float(summary.get("policy_b_mean_raw_minus_policy_b_legal_action_count", 0.0)) * summary_weight
            )
            weighted_policy_b_top_action_illegal_for_policy_a_rate += (
                float(summary.get("policy_b_top_action_illegal_for_policy_a_rate", 0.0)) * summary_weight
            )
            weighted_policy_a_top_action_illegal_for_policy_b_rate += (
                float(summary.get("policy_a_top_action_illegal_for_policy_b_rate", 0.0)) * summary_weight
            )
            for item in summary.get("top_action_family_confusions", []):
                if not isinstance(item, dict):
                    continue
                policy_b_family = str(item.get("policy_b_family", "")).strip()
                policy_a_family = str(item.get("policy_a_family", "")).strip()
                count = item.get("count")
                if policy_a_family and policy_b_family and isinstance(count, int):
                    all_step_family_confusions[(policy_b_family, policy_a_family)] += int(count)
            for item in summary.get("policy_a_mean_family_probability_masses", []):
                if not isinstance(item, dict):
                    continue
                family = str(item.get("family", "")).strip()
                probability = item.get("mean_probability")
                if family and isinstance(probability, int | float):
                    weighted_family_probability_masses[family] += float(probability) * summary_weight
        top_examples.extend(list(bundle_summary.get("top_examples", [])))
        for item in bundle_summary["family_pair_counts"]:
            family_pair_counts[(str(item["policy_a_family"]), str(item["policy_b_family"]))] += int(item["count"])
        for item in bundle_summary["policy_a_family_counts"]:
            policy_a_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["policy_b_family_counts"]:
            policy_b_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["recorded_family_counts"]:
            recorded_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["action_label_pair_counts"]:
            action_label_pair_counts[(str(item["policy_a_action_label"]), str(item["policy_b_action_label"]))] += int(
                item["count"]
            )
        for item in bundle_summary["policy_a_action_label_counts"]:
            policy_a_action_label_counts[str(item["action_label"])] += int(item["count"])
        for item in bundle_summary["policy_b_action_label_counts"]:
            policy_b_action_label_counts[str(item["action_label"])] += int(item["count"])

    top_examples.sort(key=lambda example: float(example.get("total_variation", 0.0)), reverse=True)
    mean_family_probability_masses: list[dict[str, Any]] = [
        {"family": family, "mean_probability": value / float(weighted_probability_weight)}
        for family, value in weighted_family_probability_masses.items()
        if weighted_probability_weight > 0
    ]
    mean_family_probability_masses.sort(
        key=lambda item: (-float(cast(float, item["mean_probability"])), str(item["family"]))
    )
    weighted_bundle_percentiles = weighted_bundle_percentile_summaries(bundle_summaries)

    return {
        "status": "ok" if not inspection_errors else "partial_failure",
        "policy_id": policy_id,
        "requested_policy_id": requested_policy_id or policy_id,
        "opponent_policy_id": opponent_policy_id,
        "policy_id_mismatch_allowed": bool(policy_id_mismatch_allowed),
        "source_paired_seeds_reused_for_policy_mismatch": bool(
            policy_id_mismatch_allowed and policy_id != source.focal_policy_id
        ),
        "source": {
            "run_dir": run_dir.resolve().as_posix(),
            "episodes_jsonl": episodes_jsonl.resolve().as_posix(),
            "config_hash256": source.config_hash256,
            "loaded_stack_config_hash256": stack_config_hash256,
            "run_manifest_config_hash256": run_manifest_config_hash256,
            "spec_hash256": source.spec_hash256,
            "paired_seed_count": len(source.paired_seeds),
            "paired_seeds": list(source.paired_seeds),
            "focal_policy_id": source.focal_policy_id,
            "opponent_policy_id": source.opponent_policy_id,
        },
        "output_run_dir": output_run_dir.resolve().as_posix(),
        "episodes_path": episodes_path.as_posix(),
        "replayed_game_count": int(game_count),
        "bundle_count": len(bundle_summaries),
        "games": int(game_count),
        "compared_steps": compared_steps,
        "inspected_step_count": inspected_steps,
        "max_total_variation": max_total_variation,
        "mean_total_variation": (weighted_total_variation / compared_steps if compared_steps else 0.0),
        "policy_a_matches_policy_b_top_action_rate": (
            weighted_top_action_match_rate / weighted_probability_weight if weighted_probability_weight else None
        ),
        "policy_a_matches_policy_b_top_action_family_rate": (
            weighted_top_action_family_match_rate / weighted_probability_weight if weighted_probability_weight else None
        ),
        "policy_a_mean_probability_on_policy_b_top_action": (
            weighted_probability_on_policy_b_top_action / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_a_mean_probability_on_policy_b_top_action_family": (
            weighted_probability_on_policy_b_top_action_family / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_a_weighted_mean_median_rank_of_policy_b_top_action": (
            weighted_median_rank_of_policy_b_top_action / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_a_legal_surface_filter_rate": (
            weighted_policy_a_legal_surface_filter_rate / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_b_legal_surface_filter_rate": (
            weighted_policy_b_legal_surface_filter_rate / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_a_mean_raw_minus_policy_a_legal_action_count": (
            weighted_policy_a_removed_action_count / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_b_mean_raw_minus_policy_b_legal_action_count": (
            weighted_policy_b_removed_action_count / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_b_top_action_illegal_for_policy_a_rate": (
            weighted_policy_b_top_action_illegal_for_policy_a_rate / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        "policy_a_top_action_illegal_for_policy_b_rate": (
            weighted_policy_a_top_action_illegal_for_policy_b_rate / weighted_probability_weight
            if weighted_probability_weight
            else None
        ),
        **weighted_bundle_percentiles,
        "policy_b_top_family_summaries": weighted_policy_b_top_family_summaries(bundle_summaries),
        "policy_a_mean_family_probability_masses": mean_family_probability_masses,
        "trajectory_summary": aggregate_trajectory_summary(bundle_summaries),
        "top_action_family_confusions": top_counter_items(
            all_step_family_confusions,
            key_names=("policy_b_family", "policy_a_family"),
            limit=20,
        ),
        "top_family_pairs": top_counter_items(
            family_pair_counts,
            key_names=("policy_a_family", "policy_b_family"),
        ),
        "top_policy_a_families": top_counter_items(policy_a_family_counts, key_names=("family",)),
        "top_policy_b_families": top_counter_items(policy_b_family_counts, key_names=("family",)),
        "top_recorded_families": top_counter_items(recorded_family_counts, key_names=("family",)),
        "top_action_label_pairs": top_counter_items(
            action_label_pair_counts,
            key_names=("policy_a_action_label", "policy_b_action_label"),
        ),
        "top_policy_a_action_labels": top_counter_items(
            policy_a_action_label_counts,
            key_names=("action_label",),
        ),
        "top_policy_b_action_labels": top_counter_items(
            policy_b_action_label_counts,
            key_names=("action_label",),
        ),
        "top_examples": top_examples[:5],
        "bundle_summaries": list(bundle_summaries),
        "inspection_errors": list(inspection_errors),
    }


def weighted_bundle_percentile_summaries(bundle_summaries: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        f"{source_key}_bundle_weighted": weighted_bundle_percentile_summary(
            bundle_summaries,
            source_key=source_key,
        )
        for source_key in _BUNDLE_PERCENTILE_SUMMARY_KEYS
    }


def weighted_policy_b_top_family_summaries(bundle_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for bundle_summary in bundle_summaries:
        summary = bundle_summary.get("summary")
        if not isinstance(summary, dict):
            continue
        family_summaries = summary.get("policy_b_top_family_summaries", [])
        if not isinstance(family_summaries, list):
            continue
        for raw_item in family_summaries:
            if not isinstance(raw_item, dict):
                continue
            family = str(raw_item.get("family", "")).strip()
            raw_count = raw_item.get("count")
            if not family or not isinstance(raw_count, int | float) or int(raw_count) <= 0:
                continue
            count = int(raw_count)
            target = grouped.setdefault(
                family,
                {
                    "count": 0,
                    "weighted_fields": defaultdict(float),
                    "percentile_totals": defaultdict(lambda: defaultdict(float)),
                    "percentile_weights": defaultdict(lambda: defaultdict(float)),
                    "percentile_counts": defaultdict(float),
                },
            )
            target["count"] += count
            for field in _POLICY_B_FAMILY_WEIGHTED_FIELDS:
                value = raw_item.get(field)
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    target["weighted_fields"][field] += float(value) * count
            for key in _POLICY_B_FAMILY_PERCENTILE_SUMMARY_KEYS:
                percentiles = raw_item.get(key)
                if not isinstance(percentiles, dict):
                    continue
                percentile_count = percentiles.get("count")
                if (
                    not isinstance(percentile_count, int | float)
                    or not math.isfinite(float(percentile_count))
                    or float(percentile_count) <= 0
                ):
                    continue
                weight = float(percentile_count)
                target["percentile_counts"][key] += weight
                for field in _PERCENTILE_SUMMARY_FIELDS:
                    value = percentiles.get(field)
                    if isinstance(value, int | float) and math.isfinite(float(value)):
                        target["percentile_totals"][key][field] += float(value) * weight
                        target["percentile_weights"][key][field] += weight

    payload: list[dict[str, Any]] = []
    for family, item in sorted(grouped.items(), key=lambda entry: (-int(entry[1]["count"]), entry[0])):
        count = int(item["count"])
        family_payload: dict[str, Any] = {"family": family, "count": count}
        for field in _POLICY_B_FAMILY_WEIGHTED_FIELDS:
            family_payload[field] = float(item["weighted_fields"].get(field, 0.0) / count)
        for key in _POLICY_B_FAMILY_PERCENTILE_SUMMARY_KEYS:
            summary_payload: dict[str, Any] = {
                "aggregation": "weighted_mean_of_bundle_family_percentiles",
                "source_summary_key": key,
                "count": int(item["percentile_counts"].get(key, 0.0)),
            }
            for field in _PERCENTILE_SUMMARY_FIELDS:
                weight = float(item["percentile_weights"][key].get(field, 0.0))
                summary_payload[field] = (
                    float(item["percentile_totals"][key].get(field, 0.0) / weight) if weight > 0.0 else None
                )
            family_payload[f"{key}_bundle_weighted"] = summary_payload
        payload.append(family_payload)
    return payload


def aggregate_trajectory_summary(bundle_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trajectory_summaries = [
        summary
        for bundle_summary in bundle_summaries
        if isinstance((summary := bundle_summary.get("trajectory_summary")), dict)
    ]
    aggregate = aggregate_trajectory_summary_items(trajectory_summaries)
    role_groups: dict[str, list[dict[str, Any]]] = {"focal": [], "opponent": []}
    for bundle_summary in bundle_summaries:
        focal_seat = int(bundle_summary.get("focal_seat", 0))
        trajectory_summary = bundle_summary.get("trajectory_summary")
        if not isinstance(trajectory_summary, dict):
            continue
        actor_summaries = trajectory_summary.get("actor_summaries")
        if not isinstance(actor_summaries, list):
            continue
        for actor_summary in actor_summaries:
            if not isinstance(actor_summary, dict):
                continue
            actor = actor_summary.get("actor")
            if not isinstance(actor, int):
                continue
            role = "focal" if int(actor) == focal_seat else "opponent"
            role_groups[role].append(actor_summary)
    aggregate["role_summaries"] = [
        {"role": role, **aggregate_trajectory_summary_items(items)} for role, items in role_groups.items() if items
    ]
    return aggregate


def aggregate_trajectory_summary_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    compared_steps = 0
    recorded_family_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    decision_kind_counts: Counter[str] = Counter()
    legal_family_rate_totals: defaultdict[str, float] = defaultdict(float)
    legal_family_rate_weights: defaultdict[str, float] = defaultdict(float)
    numeric_percentile_totals: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    numeric_percentile_weights: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    numeric_counts: defaultdict[str, float] = defaultdict(float)

    for item in items:
        weight = int(item.get("compared_steps", 0))
        if weight <= 0:
            continue
        compared_steps += weight
        merge_counter_payload(recorded_family_counts, item.get("recorded_family_counts"), key_name="family")
        merge_counter_payload(phase_counts, item.get("phase_counts"), key_name="phase")
        merge_counter_payload(decision_kind_counts, item.get("decision_kind_counts"), key_name="decision_kind")
        for raw_rate in item.get("legal_family_presence_rates", []):
            if not isinstance(raw_rate, dict):
                continue
            family = str(raw_rate.get("family", "")).strip()
            rate = raw_rate.get("rate")
            if family and isinstance(rate, int | float) and math.isfinite(float(rate)):
                legal_family_rate_totals[family] += float(rate) * weight
                legal_family_rate_weights[family] += weight
        numeric_summaries = item.get("numeric_summaries")
        if not isinstance(numeric_summaries, dict):
            continue
        for field, raw_summary in numeric_summaries.items():
            if not isinstance(raw_summary, dict):
                continue
            raw_count = raw_summary.get("count")
            if not isinstance(raw_count, int | float) or float(raw_count) <= 0:
                continue
            numeric_counts[str(field)] += float(raw_count)
            for percentile_field in _PERCENTILE_SUMMARY_FIELDS:
                value = raw_summary.get(percentile_field)
                if isinstance(value, int | float) and math.isfinite(float(value)):
                    numeric_percentile_totals[str(field)][percentile_field] += float(value) * float(raw_count)
                    numeric_percentile_weights[str(field)][percentile_field] += float(raw_count)

    numeric_summaries_payload: dict[str, dict[str, Any]] = {}
    for field in sorted(numeric_counts):
        field_payload: dict[str, Any] = {
            "aggregation": "weighted_mean_of_summary_percentiles",
            "count": int(numeric_counts[field]),
        }
        for percentile_field in _PERCENTILE_SUMMARY_FIELDS:
            percentile_weight = float(numeric_percentile_weights[field].get(percentile_field, 0.0))
            field_payload[percentile_field] = (
                float(numeric_percentile_totals[field].get(percentile_field, 0.0) / percentile_weight)
                if percentile_weight > 0.0
                else None
            )
        numeric_summaries_payload[field] = field_payload

    return {
        "compared_steps": int(compared_steps),
        "recorded_family_counts": counter_payload(recorded_family_counts, key_names=("family",)),
        "phase_counts": counter_payload(phase_counts, key_names=("phase",)),
        "decision_kind_counts": counter_payload(decision_kind_counts, key_names=("decision_kind",)),
        "legal_family_presence_rates": [
            {
                "family": family,
                "rate": float(legal_family_rate_totals[family] / legal_family_rate_weights[family]),
            }
            for family in sorted(legal_family_rate_weights)
            if legal_family_rate_weights[family] > 0.0
        ],
        "numeric_summaries": numeric_summaries_payload,
    }


def merge_counter_payload(counter: Counter[str], payload: Any, *, key_name: str) -> None:
    if not isinstance(payload, list):
        return
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get(key_name)
        count = item.get("count")
        if key is None or not isinstance(count, int):
            continue
        counter[str(key)] += int(count)


def weighted_bundle_percentile_summary(
    bundle_summaries: Sequence[dict[str, Any]],
    *,
    source_key: str,
) -> dict[str, Any]:
    field_totals = dict.fromkeys(_PERCENTILE_SUMMARY_FIELDS, 0.0)
    field_weights = dict.fromkeys(_PERCENTILE_SUMMARY_FIELDS, 0.0)
    total_count = 0.0

    for bundle_summary in bundle_summaries:
        summary = bundle_summary.get("summary")
        if not isinstance(summary, dict):
            continue
        percentiles = summary.get(source_key)
        if not isinstance(percentiles, dict):
            continue
        raw_count = percentiles.get("count")
        if not isinstance(raw_count, int | float) or not math.isfinite(float(raw_count)) or float(raw_count) <= 0:
            continue
        weight = float(raw_count)
        total_count += weight
        for field in _PERCENTILE_SUMMARY_FIELDS:
            raw_value = percentiles.get(field)
            if not isinstance(raw_value, int | float) or not math.isfinite(float(raw_value)):
                continue
            field_totals[field] += float(raw_value) * weight
            field_weights[field] += weight

    payload: dict[str, Any] = {
        "aggregation": "weighted_mean_of_bundle_percentiles",
        "source_summary_key": source_key,
        "count": int(total_count),
    }
    for field in _PERCENTILE_SUMMARY_FIELDS:
        weight = field_weights[field]
        payload[field] = float(field_totals[field] / weight) if weight > 0 else None
    return payload


def counter_payload(counter: Counter[Any], *, key_names: tuple[str, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, count in sorted(
        counter.items(), key=lambda item: (-int(item[1]), tuple(str(part) for part in as_tuple(item[0])))
    ):
        payload: dict[str, Any] = {"count": int(count)}
        for key_name, part in zip(key_names, as_tuple(key), strict=False):
            payload[key_name] = part
        items.append(payload)
    return items


def top_counter_items(counter: Counter[Any], *, key_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    return counter_payload(counter, key_names=key_names)[:limit]


def as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


__all__ = [
    "aggregate_audit_summary",
    "aggregate_trajectory_summary",
    "counter_payload",
    "top_counter_items",
]
