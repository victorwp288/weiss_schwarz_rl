from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TeacherActionOverrideExportConfig:
    inspection_jsons: tuple[Path, ...]
    min_total_variation: float = 0.0
    include_matches: bool = False
    max_rows_per_bundle: int | None = None
    max_rows: int | None = None


def build_teacher_action_overrides_from_inspections(
    config: TeacherActionOverrideExportConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_config(config)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for inspection_json in config.inspection_jsons:
        payload = _read_json_object(inspection_json)
        bundle_path = _bundle_path(payload, inspection_json=inspection_json)
        bundle_rows = 0
        for difference in _top_differences(payload):
            counters["candidate_rows"] += 1
            if not config.include_matches and bool(difference.get("policy_a_matches_policy_b_top_action")):
                counters["skipped_match_rows"] += 1
                continue
            total_variation = float(difference.get("total_variation", 0.0))
            if total_variation < float(config.min_total_variation):
                counters["skipped_low_total_variation_rows"] += 1
                continue
            teacher_action = _policy_b_top_action(difference)
            if teacher_action is None:
                counters["skipped_missing_teacher_action_rows"] += 1
                continue
            rows.append(
                _override_row(
                    difference,
                    bundle_path=bundle_path,
                    teacher_action=int(teacher_action),
                    source_report=inspection_json,
                )
            )
            bundle_rows += 1
            counters["written_rows"] += 1
            if config.max_rows is not None and len(rows) >= int(config.max_rows):
                break
            if config.max_rows_per_bundle is not None and bundle_rows >= int(config.max_rows_per_bundle):
                break
        if config.max_rows is not None and len(rows) >= int(config.max_rows):
            break

    summary = {
        "kind": "teacher_action_overrides_from_inspections_v1",
        "inspection_jsons": [path.as_posix() for path in config.inspection_jsons],
        "min_total_variation": float(config.min_total_variation),
        "include_matches": bool(config.include_matches),
        "max_rows_per_bundle": config.max_rows_per_bundle,
        "max_rows": config.max_rows,
        "row_count": len(rows),
        "counters": dict(sorted(counters.items())),
        "bundle_count": len({row["bundle_path"] for row in rows}),
    }
    return rows, summary


def write_teacher_action_overrides_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def write_teacher_action_overrides_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_config(config: TeacherActionOverrideExportConfig) -> None:
    if not config.inspection_jsons:
        raise ValueError("inspection_jsons must contain at least one report")
    if float(config.min_total_variation) < 0.0:
        raise ValueError("min_total_variation must be >= 0")
    if config.max_rows_per_bundle is not None and int(config.max_rows_per_bundle) <= 0:
        raise ValueError("max_rows_per_bundle must be positive when provided")
    if config.max_rows is not None and int(config.max_rows) <= 0:
        raise ValueError("max_rows must be positive when provided")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _bundle_path(payload: Mapping[str, Any], *, inspection_json: Path) -> Path:
    raw = payload.get("bundle_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"inspection report is missing bundle_path: {inspection_json}")
    return Path(raw)


def _top_differences(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw = payload.get("top_differences")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _policy_b_top_action(difference: Mapping[str, Any]) -> int | None:
    raw = difference.get("policy_b_top_action")
    if not isinstance(raw, Mapping):
        return None
    action = raw.get("action")
    return None if action is None else int(action)


def _override_row(
    difference: Mapping[str, Any],
    *,
    bundle_path: Path,
    teacher_action: int,
    source_report: Path,
) -> dict[str, Any]:
    policy_a_top = difference.get("policy_a_top_action")
    policy_b_top = difference.get("policy_b_top_action")
    return {
        "bundle_path": bundle_path.resolve().as_posix(),
        "bundle_name": bundle_path.name,
        "step_index": int(difference["step_index"]),
        "actor": int(difference.get("actor", -1)),
        "teacher_action": int(teacher_action),
        "source": "policy_b_top_action",
        "source_report": source_report.as_posix(),
        "total_variation": float(difference.get("total_variation", 0.0)),
        "policy_a_matches_policy_b_top_action": bool(difference.get("policy_a_matches_policy_b_top_action")),
        "policy_a_top_action": _action_payload(policy_a_top),
        "policy_b_top_action": _action_payload(policy_b_top),
    }


def _action_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if "action" in payload:
        payload["action"] = int(payload["action"])
    return payload


__all__ = [
    "TeacherActionOverrideExportConfig",
    "build_teacher_action_overrides_from_inspections",
    "write_teacher_action_overrides_jsonl",
    "write_teacher_action_overrides_summary",
]
