"""Replay inspection report rendering helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def format_replay_inspection_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    policy_a_source = _format_policy_source(report["policy_a"])
    policy_b_source = _format_policy_source(report["policy_b"])
    lines = [
        "Replay inspector",
        f"bundle: {report['bundle_path']}",
        f"policy_a: {report['policy_a']['label']} ({policy_a_source})",
        f"policy_b: {report['policy_b']['label']} ({policy_b_source})",
        f"compared_steps: {report['compared_steps']}",
        (
            "summary: "
            f"max_tv={summary['max_total_variation']:.6f} "
            f"mean_tv={summary['mean_total_variation']:.6f} "
            f"max_abs_prob_delta={summary['max_abs_probability_delta']:.6f}"
        ),
        (
            "alignment: "
            f"action_match_rate={summary['policy_a_matches_policy_b_top_action_rate']:.6f} "
            f"family_match_rate={summary['policy_a_matches_policy_b_top_action_family_rate']:.6f} "
            f"mean_p_on_b={summary['policy_a_mean_probability_on_policy_b_top_action']:.6f} "
            f"median_rank_of_b={summary['policy_a_median_rank_of_policy_b_top_action']:.2f}"
        ),
        "top_differences:",
    ]
    opponent_context = report.get("opponent_context")
    if isinstance(opponent_context, Mapping) and opponent_context.get("policy_id"):
        lines.insert(
            5,
            (
                "opponent_context: "
                f"policy_id={opponent_context.get('policy_id')} "
                f"policy_a_index={opponent_context.get('policy_a_index')} "
                f"policy_b_index={opponent_context.get('policy_b_index')}"
            ),
        )
    if summary["top_action_family_confusions"]:
        family_confusions = ", ".join(
            f"{item['policy_b_family']}->{item['policy_a_family']} x{item['count']}"
            for item in summary["top_action_family_confusions"]
        )
        lines.append(f"family_confusions: {family_confusions}")
    if summary.get("actor_summaries"):
        actor_lines = []
        for item in summary["actor_summaries"]:
            actor_lines.append(
                f"actor={item['actor']} steps={item['compared_steps']} "
                f"action_match={item['policy_a_matches_policy_b_top_action_rate']:.6f} "
                f"family_match={item['policy_a_matches_policy_b_top_action_family_rate']:.6f} "
                f"mean_tv={item['mean_total_variation']:.6f}"
            )
        lines.append("actor_summaries: " + "; ".join(actor_lines))
    trajectory_summary = report.get("trajectory_summary")
    if isinstance(trajectory_summary, dict) and trajectory_summary.get("recorded_family_counts"):
        family_text = ", ".join(
            f"{item['family']} x{item['count']}" for item in trajectory_summary["recorded_family_counts"][:6]
        )
        numeric = trajectory_summary.get("numeric_summaries", {})
        self_clock = numeric.get("self_clock_count", {}) if isinstance(numeric, dict) else {}
        opp_clock = numeric.get("opponent_clock_count", {}) if isinstance(numeric, dict) else {}
        lines.append(
            "trajectory: "
            f"families={family_text} "
            f"self_clock_mean={_format_optional_float(self_clock.get('mean'))} "
            f"opp_clock_mean={_format_optional_float(opp_clock.get('mean'))}"
        )

    for index, diff in enumerate(report["top_differences"], start=1):
        lines.append(
            f"{index}. step={diff['step_index']} decision_id={diff['decision_id']} actor={diff['actor']} "
            f"recorded_action={_format_action_descriptor(diff['recorded_action_detail'])} "
            f"tv={diff['total_variation']:.6f} "
            f"max_abs_prob_delta={diff['max_abs_probability_delta']:.6f}"
        )
        lines.append(
            f"   {report['policy_a']['label']}: top_action={_format_action_descriptor(diff['policy_a_top_action'])} "
            f"p={diff['policy_a_top_action']['probability']:.6f} "
            f"recorded_p={diff['policy_a_recorded_action_probability']:.6f}"
        )
        lines.append(
            f"   {report['policy_b']['label']}: top_action={_format_action_descriptor(diff['policy_b_top_action'])} "
            f"p={diff['policy_b_top_action']['probability']:.6f} "
            f"recorded_p={diff['policy_b_recorded_action_probability']:.6f}"
        )
        lines.append(
            f"   learner_on_b2: p={diff['policy_a_probability_on_policy_b_top_action']:.6f} "
            f"rank={diff['policy_a_rank_of_policy_b_top_action']} "
            f"action_match={diff['policy_a_matches_policy_b_top_action']} "
            f"family_match={diff['policy_a_matches_policy_b_top_action_family']}"
        )
        action_delta_text = ", ".join(
            (
                f"{_format_action_descriptor(item)}:delta={item['probability_delta_b_minus_a']:+.6f} "
                f"(A={item['probability_a']:.6f},B={item['probability_b']:.6f})"
            )
            for item in diff["top_action_deltas"]
        )
        lines.append(f"   deltas: {action_delta_text}")

    return "\n".join(lines) + "\n"


def write_replay_inspection_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_policy_source(policy_report: Mapping[str, Any]) -> str:
    weights_path = policy_report.get("weights_path")
    if isinstance(weights_path, str) and weights_path:
        return weights_path
    return str(policy_report.get("kind", "unknown"))


def _format_optional_float(value: Any) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.3f}"


def _format_action_descriptor(payload: Mapping[str, Any]) -> str:
    parts = [f"a{int(payload['action'])}"]
    family = payload.get("family")
    if family is None:
        return "".join(parts)
    detail_parts: list[str] = [str(family)]
    for key in ("hand_index", "stage_slot", "from_slot", "to_slot", "slot", "attack_type", "index"):
        value = payload.get(key)
        if value is not None:
            detail_parts.append(f"{key}={value}")
    return f"{parts[0]}[{', '.join(detail_parts)}]"


__all__ = [
    "format_replay_inspection_report",
    "write_replay_inspection_report",
]
