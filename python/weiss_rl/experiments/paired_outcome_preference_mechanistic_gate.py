"""Offline gate for paired-outcome preference margin diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.opponent_context_coverage import context_coverage_failures_from_report


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceMechanisticGateConfig:
    pre_report_json: Path
    post_report_json: Path
    min_mean_delta: float = 0.0
    min_min_delta: float = 0.0
    min_pair_improved_fraction: float = 1.0
    max_pair_worsened_fraction: float = 0.0
    min_group_mean_delta: float = 0.0
    min_required_group_mean_delta: float = 0.0
    required_groups: tuple[str, ...] = ()
    require_context: bool = True


def evaluate_paired_outcome_preference_mechanistic_gate(
    config: PairedOutcomePreferenceMechanisticGateConfig,
) -> dict[str, Any]:
    """Compare pre/post preference margin reports before game eval escalation."""

    pre = _read_json_object(config.pre_report_json)
    post = _read_json_object(config.post_report_json)
    pre_rows = _rows_by_pair_id(pre, label="pre")
    post_rows = _rows_by_pair_id(post, label="post")
    failures: list[str] = []
    warnings: list[str] = []

    missing_in_post = sorted(set(pre_rows) - set(post_rows))
    missing_in_pre = sorted(set(post_rows) - set(pre_rows))
    if missing_in_post:
        failures.append("missing_post_pair_ids:" + ",".join(str(item) for item in missing_in_post))
    if missing_in_pre:
        failures.append("missing_pre_pair_ids:" + ",".join(str(item) for item in missing_in_pre))

    shared_pair_ids = sorted(set(pre_rows) & set(post_rows))
    paired_rows = [_compare_pair(pre_row=pre_rows[pair_id], post_row=post_rows[pair_id]) for pair_id in shared_pair_ids]
    for row in paired_rows:
        if not bool(row["stable_identity"]):
            warnings.append(f"pair_identity_changed:{row['preference_pair_id']}")

    deltas = [float(row["delta"]) for row in paired_rows]
    pair_count = len(paired_rows)
    pair_improved = sum(1 for delta in deltas if delta > 0.0)
    pair_worsened = sum(1 for delta in deltas if delta < 0.0)
    post_positive = sum(1 for row in paired_rows if float(row["post_dpo_margin"]) > 0.0)
    pair_improved_fraction = 0.0 if pair_count == 0 else pair_improved / pair_count
    pair_worsened_fraction = 0.0 if pair_count == 0 else pair_worsened / pair_count
    mean_delta = 0.0 if not deltas else sum(deltas) / len(deltas)
    min_delta = 0.0 if not deltas else min(deltas)
    label_summaries = _label_summaries(paired_rows, required_groups=config.required_groups)

    required_group_labels = {str(item) for item in config.required_groups}
    present_group_labels = {str(item["label"]) for item in label_summaries}
    missing_groups = sorted(required_group_labels - present_group_labels)
    if missing_groups:
        failures.append("missing_required_groups:" + ",".join(missing_groups))
    for label in label_summaries:
        group_min = float(label["delta_min"])
        group_mean = float(label["delta_mean"])
        if group_mean < float(config.min_group_mean_delta):
            failures.append(
                f"group_mean_delta_below:{label['label']}:{group_mean:.6g}<{float(config.min_group_mean_delta):.6g}"
            )
        if bool(label["required"]) and group_mean < float(config.min_required_group_mean_delta):
            failures.append(
                f"required_group_mean_delta_below:{label['label']}:{group_mean:.6g}<"
                f"{float(config.min_required_group_mean_delta):.6g}"
            )
        if bool(label["required"]) and group_min < float(config.min_min_delta):
            failures.append(
                f"required_group_min_delta_below:{label['label']}:{group_min:.6g}<{float(config.min_min_delta):.6g}"
            )

    current_context_episode_count = int(post.get("current_context_episode_count") or 0)
    reference_context_episode_count = int(post.get("reference_context_episode_count") or 0)
    if config.require_context:
        failures.extend(
            context_coverage_failures_from_report(
                post,
                coverage_key="current_context_coverage",
                context_count_key="current_context_episode_count",
                prefix="current",
            )
        )
        failures.extend(
            context_coverage_failures_from_report(
                post,
                coverage_key="reference_context_coverage",
                context_count_key="reference_context_episode_count",
                prefix="reference",
            )
        )
    if pair_count <= 0:
        failures.append("empty_pair_surface")
    if mean_delta < float(config.min_mean_delta):
        failures.append(f"mean_delta_below:{mean_delta:.6g}<{float(config.min_mean_delta):.6g}")
    if min_delta < float(config.min_min_delta):
        failures.append(f"min_delta_below:{min_delta:.6g}<{float(config.min_min_delta):.6g}")
    if pair_improved_fraction < float(config.min_pair_improved_fraction):
        failures.append(
            f"pair_improved_fraction_below:{pair_improved_fraction:.6g}<{float(config.min_pair_improved_fraction):.6g}"
        )
    if pair_worsened_fraction > float(config.max_pair_worsened_fraction):
        failures.append(
            f"pair_worsened_fraction_above:{pair_worsened_fraction:.6g}>{float(config.max_pair_worsened_fraction):.6g}"
        )

    return {
        "kind": "paired_outcome_preference_mechanistic_gate_v1",
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "pre_report_json": Path(config.pre_report_json).as_posix(),
        "post_report_json": Path(config.post_report_json).as_posix(),
        "thresholds": {
            "min_mean_delta": float(config.min_mean_delta),
            "min_min_delta": float(config.min_min_delta),
            "min_pair_improved_fraction": float(config.min_pair_improved_fraction),
            "max_pair_worsened_fraction": float(config.max_pair_worsened_fraction),
            "min_group_mean_delta": float(config.min_group_mean_delta),
            "min_required_group_mean_delta": float(config.min_required_group_mean_delta),
            "required_groups": list(config.required_groups),
            "require_context": bool(config.require_context),
        },
        "summary": {
            "pair_count": pair_count,
            "current_context_episode_count": current_context_episode_count,
            "reference_context_episode_count": reference_context_episode_count,
            "current_missing_context_episode_count": _coverage_int(
                post,
                "current_context_coverage",
                "missing_context_episode_count",
            ),
            "reference_missing_context_episode_count": _coverage_int(
                post,
                "reference_context_coverage",
                "missing_context_episode_count",
            ),
            "mean_delta": mean_delta,
            "min_delta": min_delta,
            "pair_improved": pair_improved,
            "pair_worsened": pair_worsened,
            "post_positive": post_positive,
            "pair_improved_fraction": pair_improved_fraction,
            "pair_worsened_fraction": pair_worsened_fraction,
        },
        "groups": label_summaries,
    }


def write_paired_outcome_preference_mechanistic_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compare_pair(*, pre_row: Mapping[str, Any], post_row: Mapping[str, Any]) -> dict[str, Any]:
    identity_keys = (
        "preference_pair_id",
        "group_label",
        "opponent_policy_id",
        "source_pair_index",
        "preferred_label",
        "rejected_label",
    )
    stable_identity = all(pre_row.get(key) == post_row.get(key) for key in identity_keys)
    pre_margin = _safe_float(pre_row.get("dpo_margin"))
    post_margin = _safe_float(post_row.get("dpo_margin"))
    return {
        "preference_pair_id": int(post_row.get("preference_pair_id", pre_row.get("preference_pair_id"))),
        "stable_identity": stable_identity,
        "label": str(post_row.get("group_label") or pre_row.get("group_label") or ""),
        "opponent": str(post_row.get("opponent_policy_id") or pre_row.get("opponent_policy_id") or ""),
        "source_pair_index": post_row.get("source_pair_index", pre_row.get("source_pair_index")),
        "pre_dpo_margin": pre_margin,
        "post_dpo_margin": post_margin,
        "delta": post_margin - pre_margin,
        "pre_current_raw_margin": _safe_float(pre_row.get("current_raw_margin")),
        "post_current_raw_margin": _safe_float(post_row.get("current_raw_margin")),
        "pre_reference_raw_margin": _safe_float(pre_row.get("reference_raw_margin")),
        "post_reference_raw_margin": _safe_float(post_row.get("reference_raw_margin")),
    }


def _label_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_groups: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("label") or "")].append(row)
    required = {str(item) for item in required_groups}
    summaries: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        deltas = [float(row["delta"]) for row in group]
        post_margins = [float(row["post_dpo_margin"]) for row in group]
        summaries.append(
            {
                "label": label,
                "required": label in required,
                "pair_count": len(group),
                "delta_mean": sum(deltas) / max(len(deltas), 1),
                "delta_min": min(deltas) if deltas else 0.0,
                "improved_pairs": sum(1 for delta in deltas if delta > 0.0),
                "worsened_pairs": sum(1 for delta in deltas if delta < 0.0),
                "post_positive_pairs": sum(1 for margin in post_margins if margin > 0.0),
            }
        )
    return summaries


def _rows_by_pair_id(payload: Mapping[str, Any], *, label: str) -> dict[int, Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} margin report must contain rows list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pair_id = int(row.get("preference_pair_id"))
        if pair_id in result:
            raise ValueError(f"{label} margin report has duplicate preference_pair_id {pair_id}")
        result[pair_id] = row
    return result


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _coverage_int(payload: Mapping[str, Any], coverage_key: str, field_name: str) -> int:
    coverage = payload.get(coverage_key)
    if not isinstance(coverage, Mapping):
        return 0
    return int(coverage.get(field_name) or 0)


__all__ = [
    "PairedOutcomePreferenceMechanisticGateConfig",
    "evaluate_paired_outcome_preference_mechanistic_gate",
    "write_paired_outcome_preference_mechanistic_gate",
]
