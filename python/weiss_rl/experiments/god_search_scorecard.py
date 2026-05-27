from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


@dataclass(frozen=True, slots=True)
class GodSearchScorecardConfig:
    compare_jsons: tuple[Path, ...]
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    min_all_delta_wins: int = 1
    min_fixed_delta_wins: int = -2
    min_learned_delta_wins: int = 0
    max_fixed_row_drop_wins: int = 2
    max_any_row_drop_wins: int = 4


def build_god_search_scorecard(config: GodSearchScorecardConfig) -> dict[str, Any]:
    entries = [_score_compare_json(path=path, config=config) for path in config.compare_jsons]
    return {
        "kind": "god_search_scorecard_v1",
        "fixed_opponents": list(config.fixed_opponents),
        "thresholds": {
            "min_all_delta_wins": int(config.min_all_delta_wins),
            "min_fixed_delta_wins": int(config.min_fixed_delta_wins),
            "min_learned_delta_wins": int(config.min_learned_delta_wins),
            "max_fixed_row_drop_wins": int(config.max_fixed_row_drop_wins),
            "max_any_row_drop_wins": int(config.max_any_row_drop_wins),
        },
        "entries": entries,
        "counts": _count_decisions(entries),
    }


def _score_compare_json(*, path: Path, config: GodSearchScorecardConfig) -> dict[str, Any]:
    payload = _read_json_object(path)
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    groups = payload.get("groups") if isinstance(payload.get("groups"), Mapping) else {}
    paired_seeds = _infer_paired_seeds(rows)
    row_summaries = [_summarize_row(row=row, fixed_opponents=config.fixed_opponents) for row in rows]
    gate = _evaluate_loose_gate(row_summaries=row_summaries, groups=groups, config=config)
    return {
        "compare_json": path.as_posix(),
        "baseline_label": _nested_str(payload, "baseline", "label"),
        "candidate_label": _nested_str(payload, "candidate", "label"),
        "baseline_summary_json": _nested_str(payload, "baseline", "summary_json"),
        "candidate_summary_json": _nested_str(payload, "candidate", "summary_json"),
        "paired_seeds": paired_seeds,
        "group_deltas": {
            "all_delta_wins": _group_int(groups, "all_compared", "delta_wins"),
            "fixed_delta_wins": _group_int(groups, "fixed_baselines", "delta_wins"),
            "learned_delta_wins": _group_int(groups, "learned_opponents", "delta_wins"),
        },
        "rows": row_summaries,
        "loose_gate": gate,
        "escalation": _recommend_escalation(paired_seeds=paired_seeds, gate=gate),
    }


def _summarize_row(*, row: Mapping[str, Any], fixed_opponents: Sequence[str]) -> dict[str, Any]:
    opponent = str(row.get("opponent_policy_id") or "")
    delta_wins = _optional_int(row.get("delta_wins"))
    return {
        "opponent_policy_id": opponent,
        "group": "fixed" if _matches_any(opponent, fixed_opponents) else "learned",
        "delta_wins": delta_wins,
        "baseline_wins": _optional_int(row.get("baseline_wins")),
        "candidate_wins": _optional_int(row.get("candidate_wins")),
        "shared_games": _optional_int(row.get("shared_games")),
        "changed_outcome": _optional_int(row.get("changed_outcome")),
        "status": str(row.get("status") or ""),
    }


def _evaluate_loose_gate(
    *,
    row_summaries: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Any],
    config: GodSearchScorecardConfig,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    all_delta = _group_int(groups, "all_compared", "delta_wins")
    fixed_delta = _group_int(groups, "fixed_baselines", "delta_wins")
    learned_delta = _group_int(groups, "learned_opponents", "delta_wins")
    if all_delta is None or all_delta < int(config.min_all_delta_wins):
        failures.append(
            {"reason": "all_delta_too_small", "delta_wins": all_delta, "threshold": int(config.min_all_delta_wins)}
        )
    if fixed_delta is None or fixed_delta < int(config.min_fixed_delta_wins):
        failures.append(
            {
                "reason": "fixed_delta_too_small",
                "delta_wins": fixed_delta,
                "threshold": int(config.min_fixed_delta_wins),
            }
        )
    if learned_delta is None or learned_delta < int(config.min_learned_delta_wins):
        failures.append(
            {
                "reason": "learned_delta_too_small",
                "delta_wins": learned_delta,
                "threshold": int(config.min_learned_delta_wins),
            }
        )
    for row in row_summaries:
        delta = _int_value(row.get("delta_wins"))
        if row.get("group") == "fixed" and delta < -int(config.max_fixed_row_drop_wins):
            failures.append(
                {
                    "reason": "fixed_row_catastrophic_drop",
                    "opponent": row.get("opponent_policy_id"),
                    "delta_wins": delta,
                    "threshold": -int(config.max_fixed_row_drop_wins),
                }
            )
        if delta < -int(config.max_any_row_drop_wins):
            failures.append(
                {
                    "reason": "row_catastrophic_drop",
                    "opponent": row.get("opponent_policy_id"),
                    "delta_wins": delta,
                    "threshold": -int(config.max_any_row_drop_wins),
                }
            )
    return {
        "passed": not failures,
        "failures": failures,
        "all_delta_wins": all_delta,
        "fixed_delta_wins": fixed_delta,
        "learned_delta_wins": learned_delta,
    }


def _recommend_escalation(*, paired_seeds: int | None, gate: Mapping[str, Any]) -> dict[str, str]:
    if not gate.get("passed"):
        return {"decision": "stop", "reason": "loose_gate_failed"}
    if paired_seeds is None:
        return {"decision": "needs_seed_count", "reason": "could_not_infer_paired_seeds"}
    if paired_seeds < 64:
        return {"decision": "run_confirm64", "reason": "loose_gate_passed_below_confirm64"}
    if paired_seeds < 128:
        return {"decision": "run_confirm128", "reason": "confirm64_loose_gate_passed"}
    if paired_seeds < 256:
        return {"decision": "run_confirm256", "reason": "confirm128_loose_gate_passed"}
    return {"decision": "publishable_god_search_candidate", "reason": "confirm256_loose_gate_passed"}


def _infer_paired_seeds(rows: Sequence[Mapping[str, Any]]) -> int | None:
    shared_games = [
        int(row["shared_games"])
        for row in rows
        if isinstance(row.get("shared_games"), int) and int(row["shared_games"]) > 0
    ]
    if not shared_games:
        return None
    return min(shared_games) // 2


def _count_decisions(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        escalation = entry.get("escalation")
        decision = escalation.get("decision") if isinstance(escalation, Mapping) else None
        if isinstance(decision, str):
            counts[decision] = counts.get(decision, 0) + 1
    return counts


def _matches_any(value: str, candidates: Sequence[str]) -> bool:
    return any(_is_seed_wrapped_suffix_match(str(value), str(candidate)) for candidate in candidates)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _group_int(groups: Mapping[str, Any], group: str, key: str) -> int | None:
    raw_group = groups.get(group)
    if not isinstance(raw_group, Mapping):
        return None
    return _optional_int(raw_group.get(key))


def _nested_str(payload: Mapping[str, Any], group: str, key: str) -> str | None:
    raw_group = payload.get(group)
    if not isinstance(raw_group, Mapping):
        return None
    value = raw_group.get(key)
    return str(value) if isinstance(value, str) else None


def _int_value(value: object) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
