"""Gate small paired-outcome compare screens before larger main-league evals."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PairedOutcomeCompareGateConfig:
    compare_jsons: tuple[Path, ...]
    min_all_delta_wins: int = 0
    min_fixed_delta_wins: int = 0
    min_learned_delta_wins: int = 0
    max_fixed_row_drop_wins: int = 0
    max_learned_row_drop_wins: int = 0
    required_opponents: tuple[str, ...] = ()


def evaluate_paired_outcome_compare_gate(config: PairedOutcomeCompareGateConfig) -> dict[str, Any]:
    if not config.compare_jsons:
        raise ValueError("at least one paired-outcome compare JSON is required")

    entries = [_evaluate_one_compare(path=path, config=config) for path in config.compare_jsons]
    failures = [failure for entry in entries for failure in entry["failures"]]
    return {
        "kind": "paired_outcome_compare_gate_v1",
        "passed": not failures,
        "failures": failures,
        "compare_jsons": [Path(path).as_posix() for path in config.compare_jsons],
        "thresholds": {
            "min_all_delta_wins": int(config.min_all_delta_wins),
            "min_fixed_delta_wins": int(config.min_fixed_delta_wins),
            "min_learned_delta_wins": int(config.min_learned_delta_wins),
            "max_fixed_row_drop_wins": int(config.max_fixed_row_drop_wins),
            "max_learned_row_drop_wins": int(config.max_learned_row_drop_wins),
            "required_opponents": list(config.required_opponents),
        },
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "passed_count": sum(1 for entry in entries if bool(entry["passed"])),
            "failed_count": sum(1 for entry in entries if not bool(entry["passed"])),
        },
    }


def write_paired_outcome_compare_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_one_compare(*, path: Path, config: PairedOutcomeCompareGateConfig) -> dict[str, Any]:
    payload = _read_json_object(path)
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    groups = payload.get("groups") if isinstance(payload.get("groups"), Mapping) else {}
    fixed_opponents = {str(item) for item in payload.get("fixed_opponents", []) if isinstance(item, str)}
    learned_opponents = {str(item) for item in payload.get("learned_opponents", []) if isinstance(item, str)}
    present_opponents = {
        str(row.get("candidate_opponent_policy_id") or row.get("baseline_opponent_policy_id") or "") for row in rows
    }
    present_opponents = {opponent for opponent in present_opponents if opponent}

    failures: list[dict[str, Any]] = []
    group_deltas = {
        "all_delta_wins": _group_int(groups, "all_compared", "delta_wins"),
        "fixed_delta_wins": _group_int(groups, "fixed_baselines", "delta_wins"),
        "learned_delta_wins": _group_int(groups, "learned_opponents", "delta_wins"),
    }
    _check_group_delta(
        failures,
        group_name="all_compared",
        value=group_deltas["all_delta_wins"],
        threshold=int(config.min_all_delta_wins),
        reason="all_aggregate_drop",
    )
    _check_group_delta(
        failures,
        group_name="fixed_baselines",
        value=group_deltas["fixed_delta_wins"],
        threshold=int(config.min_fixed_delta_wins),
        reason="fixed_aggregate_drop",
    )
    _check_group_delta(
        failures,
        group_name="learned_opponents",
        value=group_deltas["learned_delta_wins"],
        threshold=int(config.min_learned_delta_wins),
        reason="learned_aggregate_drop",
    )

    for row in rows:
        opponent = str(row.get("candidate_opponent_policy_id") or row.get("baseline_opponent_policy_id") or "")
        delta = int(row.get("delta_wins") or 0)
        if opponent in fixed_opponents and delta < -int(config.max_fixed_row_drop_wins):
            failures.append(
                {
                    "reason": "fixed_row_drop",
                    "compare_json": Path(path).as_posix(),
                    "opponent": opponent,
                    "delta_wins": delta,
                    "threshold": -int(config.max_fixed_row_drop_wins),
                }
            )
        if opponent in learned_opponents and delta < -int(config.max_learned_row_drop_wins):
            failures.append(
                {
                    "reason": "learned_row_drop",
                    "compare_json": Path(path).as_posix(),
                    "opponent": opponent,
                    "delta_wins": delta,
                    "threshold": -int(config.max_learned_row_drop_wins),
                }
            )

    missing = [opponent for opponent in config.required_opponents if not _matches_any(opponent, present_opponents)]
    for opponent in missing:
        failures.append(
            {
                "reason": "missing_required_opponent",
                "compare_json": Path(path).as_posix(),
                "opponent": opponent,
            }
        )

    return {
        "compare_json": Path(path).as_posix(),
        "passed": not failures,
        "failures": failures,
        "baseline_label": _nested_str(payload, "baseline", "label"),
        "candidate_label": _nested_str(payload, "candidate", "label"),
        "group_deltas": group_deltas,
        "present_opponents": sorted(present_opponents),
        "fixed_opponents": sorted(fixed_opponents),
        "learned_opponents": sorted(learned_opponents),
    }


def _check_group_delta(
    failures: list[dict[str, Any]],
    *,
    group_name: str,
    value: int | None,
    threshold: int,
    reason: str,
) -> None:
    if value is None:
        failures.append({"reason": f"missing_{group_name}_delta"})
    elif int(value) < int(threshold):
        failures.append(
            {
                "reason": reason,
                "group": group_name,
                "delta_wins": int(value),
                "threshold": int(threshold),
            }
        )


def _group_int(groups: Mapping[str, Any], group_name: str, field_name: str) -> int | None:
    group = groups.get(group_name)
    if not isinstance(group, Mapping) or group.get(field_name) is None:
        return None
    return int(group[field_name])


def _matches_any(needle: str, haystack: Sequence[str] | set[str]) -> bool:
    needle_text = str(needle)
    return any(str(item) == needle_text or str(item).endswith(f"_{needle_text}") for item in haystack)


def _nested_str(payload: Mapping[str, Any], *keys: str) -> str:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return str(current) if current is not None else ""


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


__all__ = [
    "PairedOutcomeCompareGateConfig",
    "evaluate_paired_outcome_compare_gate",
    "write_paired_outcome_compare_gate",
]
