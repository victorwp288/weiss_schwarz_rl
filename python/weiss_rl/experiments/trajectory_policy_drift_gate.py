"""Gate state-matched trajectory policy drift before game eval."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryPolicyDriftGateConfig:
    drift_report_json: Path
    candidate_label: str | None = None
    max_lost_target_top_action_rate: float = 0.0
    min_gained_target_top_action_rate: float = 0.0
    min_gain_minus_loss_rate: float = 0.0
    max_top_family_changed_rate: float = 0.0
    min_mean_target_probability_delta: float = 0.0
    max_target_probability_drop: float = 0.0
    max_top_action_match_drop_rate: float = 0.0
    top_action_near_tie_margin: float | None = None
    max_lost_target_non_near_tie_rate: float | None = None
    max_top_action_changed_non_near_tie_rate: float | None = None
    require_context: bool = True


def evaluate_trajectory_policy_drift_gate(config: TrajectoryPolicyDriftGateConfig) -> dict[str, Any]:
    report = _read_json_object(config.drift_report_json)
    drift = _select_drift_summary(report, candidate_label=config.candidate_label)
    candidate_label = str(drift.get("candidate_label") or config.candidate_label or "")
    reference_label = str(drift.get("reference_label") or report.get("reference_label") or "")
    policy_summaries = {
        str(item.get("label") or ""): item for item in report.get("policy_summaries", []) if isinstance(item, Mapping)
    }
    candidate_policy = policy_summaries.get(candidate_label, {})
    reference_policy = policy_summaries.get(reference_label, {})
    failures: list[str] = []
    warnings: list[str] = []

    lost_rate = _safe_float(drift.get("lost_target_top_action_rate"))
    gained_rate = _safe_float(drift.get("gained_target_top_action_rate"))
    gain_minus_loss = gained_rate - lost_rate
    top_family_changed_rate = _safe_float(drift.get("top_family_changed_rate"))
    top_action_changed_rate = _safe_float(drift.get("top_action_changed_rate"))
    mean_probability_delta = _safe_float(drift.get("mean_target_action_probability_delta"))
    largest_probability_drop = _largest_probability_drop(drift)
    row_count = int(drift.get("row_count") or 0)
    lost_non_near_tie_rate = 0.0
    top_action_changed_non_near_tie_rate = 0.0
    if config.top_action_near_tie_margin is not None:
        margin = float(config.top_action_near_tie_margin)
        if margin < 0.0:
            raise ValueError("top_action_near_tie_margin must be nonnegative")
        lost_non_near_tie_rate = _event_non_near_tie_rate(
            drift,
            field_prefix="lost_target_top_action",
            event_rate=lost_rate,
            row_count=row_count,
            threshold=margin,
        )
        top_action_changed_non_near_tie_rate = _event_non_near_tie_rate(
            drift,
            field_prefix="top_action_changed",
            event_rate=top_action_changed_rate,
            row_count=row_count,
            threshold=margin,
        )
    candidate_top_match = _safe_float(candidate_policy.get("top_action_matches_target_rate"))
    reference_top_match = _safe_float(reference_policy.get("top_action_matches_target_rate"))
    top_match_delta = candidate_top_match - reference_top_match
    dataset_metadata = report.get("dataset_metadata") if isinstance(report.get("dataset_metadata"), Mapping) else {}
    expected_context_episodes = int(dataset_metadata.get("bundle_count") or 0)
    candidate_context_episodes = int(candidate_policy.get("opponent_context_episode_count") or 0)
    reference_context_episodes = int(reference_policy.get("opponent_context_episode_count") or 0)

    if config.require_context:
        if expected_context_episodes <= 0:
            failures.append("missing_expected_context_episode_count")
        if candidate_context_episodes < expected_context_episodes:
            failures.append(
                f"candidate_context_episodes_below:{candidate_context_episodes}<{expected_context_episodes}"
            )
        if reference_context_episodes < expected_context_episodes:
            failures.append(
                f"reference_context_episodes_below:{reference_context_episodes}<{expected_context_episodes}"
            )
    if lost_rate > float(config.max_lost_target_top_action_rate):
        failures.append(
            f"lost_target_top_action_rate_above:{lost_rate:.6g}>{float(config.max_lost_target_top_action_rate):.6g}"
        )
    if gained_rate < float(config.min_gained_target_top_action_rate):
        failures.append(
            f"gained_target_top_action_rate_below:{gained_rate:.6g}<"
            f"{float(config.min_gained_target_top_action_rate):.6g}"
        )
    if gain_minus_loss < float(config.min_gain_minus_loss_rate):
        failures.append(
            f"gain_minus_loss_rate_below:{gain_minus_loss:.6g}<{float(config.min_gain_minus_loss_rate):.6g}"
        )
    if top_family_changed_rate > float(config.max_top_family_changed_rate):
        failures.append(
            f"top_family_changed_rate_above:{top_family_changed_rate:.6g}>"
            f"{float(config.max_top_family_changed_rate):.6g}"
        )
    if mean_probability_delta < float(config.min_mean_target_probability_delta):
        failures.append(
            f"mean_target_probability_delta_below:{mean_probability_delta:.6g}<"
            f"{float(config.min_mean_target_probability_delta):.6g}"
        )
    if largest_probability_drop < -float(config.max_target_probability_drop):
        failures.append(
            f"largest_target_probability_drop_below:{largest_probability_drop:.6g}<-"
            f"{float(config.max_target_probability_drop):.6g}"
        )
    if top_match_delta < -float(config.max_top_action_match_drop_rate):
        failures.append(
            f"top_action_match_delta_below:{top_match_delta:.6g}<-{float(config.max_top_action_match_drop_rate):.6g}"
        )
    if config.max_lost_target_non_near_tie_rate is not None:
        if lost_non_near_tie_rate > float(config.max_lost_target_non_near_tie_rate):
            failures.append(
                f"lost_target_non_near_tie_rate_above:{lost_non_near_tie_rate:.6g}>"
                f"{float(config.max_lost_target_non_near_tie_rate):.6g}"
            )
    if config.max_top_action_changed_non_near_tie_rate is not None:
        if top_action_changed_non_near_tie_rate > float(config.max_top_action_changed_non_near_tie_rate):
            failures.append(
                f"top_action_changed_non_near_tie_rate_above:{top_action_changed_non_near_tie_rate:.6g}>"
                f"{float(config.max_top_action_changed_non_near_tie_rate):.6g}"
            )
    if not candidate_label:
        warnings.append("empty_candidate_label")
    if not reference_label:
        warnings.append("empty_reference_label")

    return {
        "kind": "trajectory_policy_drift_gate_v1",
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "drift_report_json": Path(config.drift_report_json).as_posix(),
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "thresholds": {
            "max_lost_target_top_action_rate": float(config.max_lost_target_top_action_rate),
            "min_gained_target_top_action_rate": float(config.min_gained_target_top_action_rate),
            "min_gain_minus_loss_rate": float(config.min_gain_minus_loss_rate),
            "max_top_family_changed_rate": float(config.max_top_family_changed_rate),
            "min_mean_target_probability_delta": float(config.min_mean_target_probability_delta),
            "max_target_probability_drop": float(config.max_target_probability_drop),
            "max_top_action_match_drop_rate": float(config.max_top_action_match_drop_rate),
            "top_action_near_tie_margin": None
            if config.top_action_near_tie_margin is None
            else float(config.top_action_near_tie_margin),
            "max_lost_target_non_near_tie_rate": None
            if config.max_lost_target_non_near_tie_rate is None
            else float(config.max_lost_target_non_near_tie_rate),
            "max_top_action_changed_non_near_tie_rate": None
            if config.max_top_action_changed_non_near_tie_rate is None
            else float(config.max_top_action_changed_non_near_tie_rate),
            "require_context": bool(config.require_context),
        },
        "summary": {
            "row_count": row_count,
            "expected_context_episodes": expected_context_episodes,
            "candidate_context_episodes": candidate_context_episodes,
            "reference_context_episodes": reference_context_episodes,
            "lost_target_top_action_rate": lost_rate,
            "gained_target_top_action_rate": gained_rate,
            "gain_minus_loss_rate": gain_minus_loss,
            "top_action_changed_rate": top_action_changed_rate,
            "lost_target_non_near_tie_rate": lost_non_near_tie_rate,
            "top_action_changed_non_near_tie_rate": top_action_changed_non_near_tie_rate,
            "top_family_changed_rate": top_family_changed_rate,
            "mean_target_action_probability_delta": mean_probability_delta,
            "largest_target_probability_drop": largest_probability_drop,
            "candidate_top_action_matches_target_rate": candidate_top_match,
            "reference_top_action_matches_target_rate": reference_top_match,
            "top_action_match_delta": top_match_delta,
        },
    }


def write_trajectory_policy_drift_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_drift_summary(report: Mapping[str, Any], *, candidate_label: str | None) -> Mapping[str, Any]:
    summaries = report.get("drift_summaries")
    if not isinstance(summaries, list):
        raise ValueError("trajectory policy drift report must contain drift_summaries")
    candidates = [item for item in summaries if isinstance(item, Mapping)]
    if candidate_label is not None:
        matching = [item for item in candidates if str(item.get("candidate_label") or "") == str(candidate_label)]
        if not matching:
            raise ValueError(f"candidate label not found in drift report: {candidate_label}")
        return matching[0]
    if len(candidates) != 1:
        raise ValueError("candidate_label is required when drift report has zero or multiple candidate summaries")
    return candidates[0]


def _largest_probability_drop(drift: Mapping[str, Any]) -> float:
    drops = drift.get("largest_target_probability_drops")
    if not isinstance(drops, list) or not drops:
        return 0.0
    values = [
        _safe_float(item.get("probability_delta"))
        for item in drops
        if isinstance(item, Mapping) and item.get("probability_delta") is not None
    ]
    return min(values) if values else 0.0


def _event_non_near_tie_rate(
    drift: Mapping[str, Any],
    *,
    field_prefix: str,
    event_rate: float,
    row_count: int,
    threshold: float,
) -> float:
    if row_count <= 0:
        return 0.0
    event_count = int(round(float(event_rate) * float(row_count)))
    if event_count <= 0:
        return 0.0
    near_tie_count = _near_tie_count(drift, field_prefix=field_prefix, threshold=threshold)
    non_near_tie_count = max(0, event_count - near_tie_count)
    return float(non_near_tie_count) / float(row_count)


def _near_tie_count(drift: Mapping[str, Any], *, field_prefix: str, threshold: float) -> int:
    margin_summary = drift.get(f"{field_prefix}_candidate_top_over_target_margin")
    if not isinstance(margin_summary, Mapping):
        return 0
    near_tie_thresholds = margin_summary.get("near_tie_thresholds")
    if not isinstance(near_tie_thresholds, list):
        return 0
    tolerance = max(1e-12, abs(float(threshold)) * 1e-9)
    for item in near_tie_thresholds:
        if not isinstance(item, Mapping):
            continue
        raw_threshold = item.get("threshold")
        if raw_threshold is None:
            continue
        if abs(float(raw_threshold) - float(threshold)) <= tolerance:
            return int(item.get("count") or 0)
    return 0


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


__all__ = [
    "TrajectoryPolicyDriftGateConfig",
    "evaluate_trajectory_policy_drift_gate",
    "write_trajectory_policy_drift_gate",
]
