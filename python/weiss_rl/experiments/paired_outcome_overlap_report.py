"""Report same-pair overlaps between fixed and learned paired-outcome flips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WIN_OUTCOME = "W"


@dataclass(frozen=True)
class PairedOutcomeOverlapReportConfig:
    compare_json_paths: tuple[Path, ...]
    max_examples_per_key: int = 20


def build_paired_outcome_overlap_report(config: PairedOutcomeOverlapReportConfig) -> dict[str, Any]:
    if not config.compare_json_paths:
        raise ValueError("at least one compare_json_path is required")
    if int(config.max_examples_per_key) < 1:
        raise ValueError("max_examples_per_key must be >= 1")

    reports = []
    for path in config.compare_json_paths:
        reports.append(_build_single_report(path=Path(path), max_examples_per_key=int(config.max_examples_per_key)))
    return {
        "kind": "paired_outcome_overlap_report_v1",
        "report_count": len(reports),
        "reports": reports,
        "total_conflict_key_count": sum(int(report["conflict_key_count"]) for report in reports),
        "total_truncated_rows": sum(int(report["truncated_row_count"]) for report in reports),
    }


def write_paired_outcome_overlap_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_single_report(*, path: Path, max_examples_per_key: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixed_opponents = {str(item) for item in payload.get("fixed_opponents", [])}
    learned_opponents = {str(item) for item in payload.get("learned_opponents", [])}
    candidate_label = str((payload.get("candidate") or {}).get("label") or "candidate")
    baseline_label = str((payload.get("baseline") or {}).get("label") or "baseline")

    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    row_summaries = []
    truncated_rows = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        opponent = str(row.get("opponent_policy_id") or row.get("opponent") or "")
        if not opponent:
            continue
        panel = _classify_panel(opponent, fixed_opponents=fixed_opponents, learned_opponents=learned_opponents)
        changed_outcome = int(row.get("changed_outcome") or 0)
        examples = _normalize_examples(row.get("examples"))
        truncated = changed_outcome > len(examples)
        if truncated:
            truncated_rows.append(
                {
                    "opponent_policy_id": opponent,
                    "changed_outcome": changed_outcome,
                    "example_count": len(examples),
                    "missing_examples": changed_outcome - len(examples),
                }
            )
        row_summaries.append(
            {
                "opponent_policy_id": opponent,
                "panel": panel,
                "delta_wins": int(row.get("delta_wins") or 0),
                "changed_outcome": changed_outcome,
                "example_count": len(examples),
                "truncated_examples": truncated,
            }
        )
        for example in examples:
            event = _event_from_example(example=example, opponent=opponent, panel=panel)
            if event is None:
                continue
            key = (
                int(event["pair_index"]),
                int(event["swap_index"]),
                int(event["episode_seed"]),
            )
            events_by_key.setdefault(key, []).append(event)

    key_summaries = [
        _summarize_key(key=key, events=events, max_examples_per_key=max_examples_per_key)
        for key, events in sorted(events_by_key.items())
    ]
    conflict_keys = [item for item in key_summaries if item["conflict_types"]]
    return {
        "compare_json": path.as_posix(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "row_count": len(row_summaries),
        "rows": row_summaries,
        "changed_event_count": sum(len(events) for events in events_by_key.values()),
        "distinct_change_key_count": len(key_summaries),
        "conflict_key_count": len(conflict_keys),
        "conflict_keys": sorted(
            conflict_keys,
            key=lambda item: (
                -len(item["events"]),
                int(item["pair_index"]),
                int(item["swap_index"]),
                int(item["episode_seed"]),
            ),
        ),
        "key_summaries": key_summaries,
        "truncated_row_count": len(truncated_rows),
        "truncated_rows": truncated_rows,
    }


def _classify_panel(opponent: str, *, fixed_opponents: set[str], learned_opponents: set[str]) -> str:
    if opponent in fixed_opponents:
        return "fixed"
    if not learned_opponents or opponent in learned_opponents:
        return "learned"
    return "other"


def _normalize_examples(raw_examples: Any) -> list[dict[str, Any]]:
    if raw_examples is None:
        return []
    if isinstance(raw_examples, dict):
        raw_examples = [raw_examples]
    if not isinstance(raw_examples, list):
        return []
    return [item for item in raw_examples if isinstance(item, dict)]


def _event_from_example(*, example: dict[str, Any], opponent: str, panel: str) -> dict[str, Any] | None:
    baseline_outcome = str(example.get("baseline_outcome") or "")
    candidate_outcome = str(example.get("candidate_outcome") or "")
    direction = _direction(baseline_outcome=baseline_outcome, candidate_outcome=candidate_outcome)
    if direction == "unchanged":
        return None
    try:
        pair_index = int(example["pair_index"])
        swap_index = int(example["swap_index"])
        episode_seed = int(example["episode_seed"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "opponent_policy_id": opponent,
        "panel": panel,
        "direction": direction,
        "pair_index": pair_index,
        "swap_index": swap_index,
        "episode_seed": episode_seed,
        "baseline_outcome": baseline_outcome,
        "candidate_outcome": candidate_outcome,
        "decision_count_delta": _optional_int(example.get("candidate_decision_count"))
        - _optional_int(example.get("baseline_decision_count")),
        "pass_actions_delta": _optional_int(example.get("candidate_pass_actions"))
        - _optional_int(example.get("baseline_pass_actions")),
        "pass_with_nonpass_available_delta": _optional_int(example.get("candidate_pass_with_nonpass_available"))
        - _optional_int(example.get("baseline_pass_with_nonpass_available")),
    }


def _direction(*, baseline_outcome: str, candidate_outcome: str) -> str:
    if baseline_outcome == WIN_OUTCOME and candidate_outcome != WIN_OUTCOME:
        return "candidate_loss"
    if baseline_outcome != WIN_OUTCOME and candidate_outcome == WIN_OUTCOME:
        return "candidate_gain"
    if baseline_outcome != candidate_outcome:
        return "other_changed"
    return "unchanged"


def _optional_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summarize_key(
    *,
    key: tuple[int, int, int],
    events: list[dict[str, Any]],
    max_examples_per_key: int,
) -> dict[str, Any]:
    counts = {
        "fixed_candidate_gain": 0,
        "fixed_candidate_loss": 0,
        "learned_candidate_gain": 0,
        "learned_candidate_loss": 0,
        "other_candidate_gain": 0,
        "other_candidate_loss": 0,
        "other_changed": 0,
    }
    for event in events:
        panel = str(event["panel"])
        direction = str(event["direction"])
        if direction == "other_changed":
            counts["other_changed"] += 1
            continue
        key_name = f"{panel}_{direction}"
        if key_name in counts:
            counts[key_name] += 1

    conflict_types = []
    if counts["fixed_candidate_loss"] > 0 and counts["learned_candidate_gain"] > 0:
        conflict_types.append("fixed_loss_and_learned_gain")
    if counts["fixed_candidate_gain"] > 0 and counts["learned_candidate_loss"] > 0:
        conflict_types.append("fixed_gain_and_learned_loss")
    if counts["fixed_candidate_gain"] > 0 and counts["fixed_candidate_loss"] > 0:
        conflict_types.append("mixed_fixed_directions")
    if counts["learned_candidate_gain"] > 0 and counts["learned_candidate_loss"] > 0:
        conflict_types.append("mixed_learned_directions")

    pair_index, swap_index, episode_seed = key
    return {
        "pair_index": pair_index,
        "swap_index": swap_index,
        "episode_seed": episode_seed,
        "event_count": len(events),
        "counts": counts,
        "conflict_types": conflict_types,
        "events": sorted(events, key=lambda item: (str(item["panel"]), str(item["opponent_policy_id"])))[
            :max_examples_per_key
        ],
        "events_truncated": len(events) > max_examples_per_key,
    }
