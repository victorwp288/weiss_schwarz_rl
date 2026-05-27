"""Offline gate for paired-swing mechanistic diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.opponent_context_coverage import context_coverage_failures_from_report


@dataclass(frozen=True, slots=True)
class PairedSwingMechanisticGateConfig:
    pre_report_json: Path
    post_report_json: Path
    min_mean_delta: float = 0.0
    min_min_delta: float = 0.0
    min_row_improved_fraction: float = 0.60
    max_row_worsened_fraction: float = 0.15
    min_top_positive_delta: int = 0
    max_positive_rank_worsened_fraction: float = 0.05
    min_protected_label_mean_delta: float = 0.0
    protected_label_contains: tuple[str, ...] = ("preserve",)
    require_context: bool = True


def evaluate_paired_swing_mechanistic_gate(config: PairedSwingMechanisticGateConfig) -> dict[str, Any]:
    """Compare pre/post margin reports and return a pass/fail gate summary."""

    pre = _read_json_object(config.pre_report_json)
    post = _read_json_object(config.post_report_json)
    pre_rows = _rows(pre, label="pre")
    post_rows = _rows(post, label="post")
    failures: list[str] = []
    warnings: list[str] = []
    if len(pre_rows) != len(post_rows):
        failures.append(f"row_count_mismatch:{len(pre_rows)}!={len(post_rows)}")
    row_count = min(len(pre_rows), len(post_rows))
    paired_rows = [
        _compare_row(pre_row=pre_rows[index], post_row=post_rows[index], index=index) for index in range(row_count)
    ]
    for row in paired_rows:
        if not bool(row["stable_identity"]):
            warnings.append(f"row_identity_changed:{row['index']}")

    row_deltas = [float(row["delta"]) for row in paired_rows]
    row_improved = sum(1 for delta in row_deltas if delta > 0.0)
    row_worsened = sum(1 for delta in row_deltas if delta < 0.0)
    row_nonnegative_post = sum(1 for row in paired_rows if float(row["post_margin"]) >= 0.0)
    row_improved_fraction = 0.0 if row_count == 0 else row_improved / row_count
    row_worsened_fraction = 0.0 if row_count == 0 else row_worsened / row_count

    pre_top_positive = sum(1 for row in paired_rows if _top_is_positive(row, side="pre"))
    post_top_positive = sum(1 for row in paired_rows if _top_is_positive(row, side="post"))
    top_positive_delta = post_top_positive - pre_top_positive
    top_action_flips = sum(1 for row in paired_rows if row.get("pre_top_action") != row.get("post_top_action"))
    positive_rank_improved = sum(1 for row in paired_rows if _rank_delta(row) < 0)
    positive_rank_worsened = sum(1 for row in paired_rows if _rank_delta(row) > 0)
    positive_rank_worsened_fraction = 0.0 if row_count == 0 else positive_rank_worsened / row_count
    missing_decision_field_rows = sum(1 for row in paired_rows if _missing_decision_fields(row))

    label_summaries = _label_summaries(paired_rows, config.protected_label_contains)
    protected_failures = [
        label
        for label in label_summaries
        if bool(label["protected"]) and float(label["delta_mean"]) < float(config.min_protected_label_mean_delta)
    ]

    mean_delta = _safe_float(post.get("positive_margin_mean")) - _safe_float(pre.get("positive_margin_mean"))
    min_delta = _safe_float(post.get("positive_margin_min")) - _safe_float(pre.get("positive_margin_min"))
    context_episode_count = int(post.get("context_episode_count") or 0)
    if config.require_context:
        failures.extend(
            context_coverage_failures_from_report(
                post,
                coverage_key="context_coverage",
                context_count_key="context_episode_count",
            )
        )
    if row_count <= 0:
        failures.append("empty_row_surface")
    if missing_decision_field_rows > 0:
        failures.append(f"missing_decision_fields:{missing_decision_field_rows}")
    if mean_delta < float(config.min_mean_delta):
        failures.append(f"mean_delta_below:{mean_delta:.6g}<{float(config.min_mean_delta):.6g}")
    if min_delta < float(config.min_min_delta):
        failures.append(f"min_delta_below:{min_delta:.6g}<{float(config.min_min_delta):.6g}")
    if row_improved_fraction < float(config.min_row_improved_fraction):
        failures.append(
            f"row_improved_fraction_below:{row_improved_fraction:.6g}<{float(config.min_row_improved_fraction):.6g}"
        )
    if row_worsened_fraction > float(config.max_row_worsened_fraction):
        failures.append(
            f"row_worsened_fraction_above:{row_worsened_fraction:.6g}>{float(config.max_row_worsened_fraction):.6g}"
        )
    if top_positive_delta < int(config.min_top_positive_delta):
        failures.append(f"top_positive_delta_below:{top_positive_delta}<{int(config.min_top_positive_delta)}")
    if positive_rank_worsened_fraction > float(config.max_positive_rank_worsened_fraction):
        failures.append(
            f"positive_rank_worsened_fraction_above:{positive_rank_worsened_fraction:.6g}>"
            f"{float(config.max_positive_rank_worsened_fraction):.6g}"
        )
    if protected_failures:
        failures.append("protected_label_mean_drop:" + ",".join(str(item["label"]) for item in protected_failures))

    return {
        "kind": "paired_swing_mechanistic_gate_v1",
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "pre_report_json": Path(config.pre_report_json).as_posix(),
        "post_report_json": Path(config.post_report_json).as_posix(),
        "thresholds": {
            "min_mean_delta": float(config.min_mean_delta),
            "min_min_delta": float(config.min_min_delta),
            "min_row_improved_fraction": float(config.min_row_improved_fraction),
            "max_row_worsened_fraction": float(config.max_row_worsened_fraction),
            "min_top_positive_delta": int(config.min_top_positive_delta),
            "max_positive_rank_worsened_fraction": float(config.max_positive_rank_worsened_fraction),
            "min_protected_label_mean_delta": float(config.min_protected_label_mean_delta),
            "protected_label_contains": list(config.protected_label_contains),
            "require_context": bool(config.require_context),
        },
        "summary": {
            "row_count": row_count,
            "context_episode_count": context_episode_count,
            "missing_context_episode_count": _coverage_int(post, "missing_context_episode_count"),
            "mean_delta": mean_delta,
            "min_delta": min_delta,
            "row_improved": row_improved,
            "row_worsened": row_worsened,
            "row_nonnegative_post": row_nonnegative_post,
            "row_improved_fraction": row_improved_fraction,
            "row_worsened_fraction": row_worsened_fraction,
            "pre_top_is_positive": pre_top_positive,
            "post_top_is_positive": post_top_positive,
            "top_positive_delta": top_positive_delta,
            "top_action_flips": top_action_flips,
            "positive_rank_improved": positive_rank_improved,
            "positive_rank_worsened": positive_rank_worsened,
            "positive_rank_worsened_fraction": positive_rank_worsened_fraction,
            "missing_decision_field_rows": missing_decision_field_rows,
        },
        "labels": label_summaries,
    }


def write_paired_swing_mechanistic_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compare_row(*, pre_row: Mapping[str, Any], post_row: Mapping[str, Any], index: int) -> dict[str, Any]:
    identity_keys = (
        "step_index",
        "episode_index",
        "source_dataset_label",
        "source_opponent_policy_id",
        "positive_action",
        "negative_action",
    )
    stable_identity = all(pre_row.get(key) == post_row.get(key) for key in identity_keys)
    pre_margin = _safe_float(pre_row.get("positive_minus_negative_logp"))
    post_margin = _safe_float(post_row.get("positive_minus_negative_logp"))
    return {
        "index": int(index),
        "stable_identity": stable_identity,
        "label": str(post_row.get("source_dataset_label") or pre_row.get("source_dataset_label") or ""),
        "opponent": str(post_row.get("source_opponent_policy_id") or pre_row.get("source_opponent_policy_id") or ""),
        "positive_action": _optional_int(post_row.get("positive_action", pre_row.get("positive_action"))),
        "negative_action": _optional_int(post_row.get("negative_action", pre_row.get("negative_action"))),
        "pre_margin": pre_margin,
        "post_margin": post_margin,
        "delta": post_margin - pre_margin,
        "pre_top_action": _optional_int(pre_row.get("top_action")),
        "post_top_action": _optional_int(post_row.get("top_action")),
        "pre_positive_rank": _optional_int(pre_row.get("positive_rank")),
        "post_positive_rank": _optional_int(post_row.get("positive_rank")),
    }


def _label_summaries(rows: Sequence[Mapping[str, Any]], protected_substrings: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("label") or "")].append(row)
    summaries: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        deltas = [float(row["delta"]) for row in group]
        protected = any(fragment and fragment in label for fragment in protected_substrings)
        summaries.append(
            {
                "label": label,
                "protected": protected,
                "row_count": len(group),
                "delta_mean": sum(deltas) / max(len(deltas), 1),
                "improved_rows": sum(1 for delta in deltas if delta > 0.0),
                "worsened_rows": sum(1 for delta in deltas if delta < 0.0),
                "post_top_is_positive": sum(1 for row in group if _top_is_positive(row, side="post")),
                "positive_rank_worsened": sum(1 for row in group if _rank_delta(row) > 0),
            }
        )
    return summaries


def _top_is_positive(row: Mapping[str, Any], *, side: str) -> bool:
    return row.get(f"{side}_top_action") is not None and row.get(f"{side}_top_action") == row.get("positive_action")


def _rank_delta(row: Mapping[str, Any]) -> int:
    pre = row.get("pre_positive_rank")
    post = row.get("post_positive_rank")
    if pre is None or post is None:
        return 0
    return int(post) - int(pre)


def _missing_decision_fields(row: Mapping[str, Any]) -> bool:
    return any(
        row.get(key) is None for key in ("pre_top_action", "post_top_action", "pre_positive_rank", "post_positive_rank")
    )


def _rows(payload: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} margin report must contain rows list")
    return [row for row in rows if isinstance(row, Mapping)]


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _coverage_int(payload: Mapping[str, Any], field_name: str) -> int:
    coverage = payload.get("context_coverage")
    if not isinstance(coverage, Mapping):
        return 0
    return int(coverage.get(field_name) or 0)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "PairedSwingMechanisticGateConfig",
    "evaluate_paired_swing_mechanistic_gate",
    "write_paired_swing_mechanistic_gate",
]
