"""Aggregation helpers for the B2 disagreement audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from weiss_rl.diagnostics.b2_audit.b2_audit_reports import b2_audit_plan_payload
from weiss_rl.diagnostics.b2_audit.b2_audit_summary_math import (
    aggregate_trajectory_summary,
    counter_payload,
    top_counter_items,
    weighted_bundle_percentile_summaries,
    weighted_policy_b_top_family_summaries,
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
        "audit_plan": b2_audit_plan_payload(),
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


__all__ = [
    "aggregate_audit_summary",
    "aggregate_trajectory_summary",
    "counter_payload",
    "top_counter_items",
]
