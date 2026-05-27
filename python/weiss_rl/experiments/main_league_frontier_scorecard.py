from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS

MAIN_LEAGUE_SENTINEL_OPPONENTS = (
    "B2 HeuristicPublic",
    "B4 HeuristicPublicControl",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
)

MAIN_LEAGUE_LEARNED_OPPONENTS = (
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000001",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000002",
    "seed_b8c698d26a_seed_c3aac2f9dc_checkpoint_000025",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_bestresponse_u25_devbest",
    "seed_b8c698d26a_seed_c3aac2f9dc_main_league_selected",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000003",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000004",
    "seed_b8c698d26a_seed_c3aac2f9dc_policy_000005",
)

MAIN_LEAGUE_FULL13_OPPONENTS = (*FIXED_THESIS_OPPONENTS, *MAIN_LEAGUE_LEARNED_OPPONENTS)


@dataclass(frozen=True, slots=True)
class MainLeagueFrontierScorecardConfig:
    compare_jsons: tuple[Path, ...]
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    sentinel_opponents: tuple[str, ...] = MAIN_LEAGUE_SENTINEL_OPPONENTS
    min_sentinel_learned_delta_wins: int = 0
    max_sentinel_fixed_row_drop_wins: int = 0
    max_sentinel_learned_row_drop_wins: int = 0
    min_full_fixed_delta_wins: int = 0
    min_full_learned_delta_wins: int = 0
    max_full_fixed_row_drop_wins: int = 0
    max_confirm128_learned_row_drop_wins: int = 0
    max_confirm256_learned_row_drop_wins: int = 0


def build_main_league_frontier_scorecard(config: MainLeagueFrontierScorecardConfig) -> dict[str, Any]:
    entries = [_summarize_compare_report(path=path, config=config) for path in config.compare_jsons]
    return {
        "kind": "main_league_frontier_scorecard_v1",
        "fixed_opponents": list(config.fixed_opponents),
        "sentinel_opponents": list(config.sentinel_opponents),
        "thresholds": {
            "min_sentinel_learned_delta_wins": int(config.min_sentinel_learned_delta_wins),
            "max_sentinel_fixed_row_drop_wins": int(config.max_sentinel_fixed_row_drop_wins),
            "max_sentinel_learned_row_drop_wins": int(config.max_sentinel_learned_row_drop_wins),
            "min_full_fixed_delta_wins": int(config.min_full_fixed_delta_wins),
            "min_full_learned_delta_wins": int(config.min_full_learned_delta_wins),
            "max_full_fixed_row_drop_wins": int(config.max_full_fixed_row_drop_wins),
            "max_confirm128_learned_row_drop_wins": int(config.max_confirm128_learned_row_drop_wins),
            "max_confirm256_learned_row_drop_wins": int(config.max_confirm256_learned_row_drop_wins),
        },
        "entries": entries,
        "counts": _count_entries(entries),
    }


def _summarize_compare_report(*, path: Path, config: MainLeagueFrontierScorecardConfig) -> dict[str, Any]:
    payload = _read_json_object(path)
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    groups = payload.get("groups") if isinstance(payload.get("groups"), Mapping) else {}
    fixed_opponents = tuple(config.fixed_opponents)
    learned_opponents = tuple(str(item) for item in payload.get("learned_opponents", []) if isinstance(item, str))
    row_summaries = [
        _summarize_row(row=row, fixed_opponents=fixed_opponents, sentinel_opponents=config.sentinel_opponents)
        for row in rows
    ]
    paired_seeds = _infer_paired_seeds(row_summaries)
    panel_kind = _classify_panel(
        row_summaries=row_summaries,
        fixed_opponents=config.fixed_opponents,
        sentinel_opponents=config.sentinel_opponents,
    )
    sentinel_gate = _evaluate_sentinel_gate(
        row_summaries=row_summaries,
        config=config,
    )
    full_gate = _evaluate_full_gate(
        row_summaries=row_summaries,
        groups=groups,
        paired_seeds=paired_seeds,
        config=config,
    )
    escalation = _recommend_escalation(
        panel_kind=panel_kind,
        paired_seeds=paired_seeds,
        sentinel_gate=sentinel_gate,
        full_gate=full_gate,
    )
    return {
        "compare_json": path.as_posix(),
        "baseline_label": _nested_str(payload, "baseline", "label"),
        "candidate_label": _nested_str(payload, "candidate", "label"),
        "baseline_summary_json": _nested_str(payload, "baseline", "summary_json"),
        "candidate_summary_json": _nested_str(payload, "candidate", "summary_json"),
        "paired_seeds": paired_seeds,
        "panel_kind": panel_kind,
        "group_deltas": {
            "all_delta_wins": _group_int(groups, "all_compared", "delta_wins"),
            "fixed_delta_wins": _group_int(groups, "fixed_baselines", "delta_wins"),
            "learned_delta_wins": _group_int(groups, "learned_opponents", "delta_wins"),
            "all_changed_outcome": _group_int(groups, "all_compared", "changed_outcome"),
            "fixed_changed_outcome": _group_int(groups, "fixed_baselines", "changed_outcome"),
            "learned_changed_outcome": _group_int(groups, "learned_opponents", "changed_outcome"),
        },
        "learned_opponents": list(learned_opponents),
        "rows": row_summaries,
        "row_regressions": {
            "fixed": [
                row for row in row_summaries if row["group"] == "fixed" and _int_value(row.get("delta_wins")) < 0
            ],
            "learned": [
                row for row in row_summaries if row["group"] == "learned" and _int_value(row.get("delta_wins")) < 0
            ],
            "sentinel": [row for row in row_summaries if row["is_sentinel"] and _int_value(row.get("delta_wins")) < 0],
        },
        "sentinel_gate": sentinel_gate,
        "full_gate": full_gate,
        "escalation": escalation,
    }


def _summarize_row(
    *,
    row: Mapping[str, Any],
    fixed_opponents: Sequence[str],
    sentinel_opponents: Sequence[str],
) -> dict[str, Any]:
    opponent = str(row.get("opponent_policy_id") or "")
    group = "fixed" if _matches_any(opponent, fixed_opponents) else "learned"
    delta_wins = _optional_int(row.get("delta_wins"))
    shared_games = _optional_int(row.get("shared_games"))
    return {
        "opponent_policy_id": opponent,
        "group": group,
        "is_sentinel": _matches_any(opponent, sentinel_opponents),
        "status": str(row.get("status") or ""),
        "baseline_wins": _optional_int(row.get("baseline_wins")),
        "candidate_wins": _optional_int(row.get("candidate_wins")),
        "delta_wins": delta_wins,
        "shared_games": shared_games,
        "candidate_mean": _optional_float(row.get("candidate_mean")),
        "baseline_mean": _optional_float(row.get("baseline_mean")),
        "changed_outcome": _optional_int(row.get("changed_outcome")),
    }


def _evaluate_sentinel_gate(
    *,
    row_summaries: Sequence[Mapping[str, Any]],
    config: MainLeagueFrontierScorecardConfig,
) -> dict[str, Any]:
    sentinel_rows = [row for row in row_summaries if bool(row.get("is_sentinel"))]
    fixed_rows = [row for row in sentinel_rows if row.get("group") == "fixed"]
    learned_rows = [row for row in sentinel_rows if row.get("group") == "learned"]
    failures: list[dict[str, Any]] = []
    present = {
        row["opponent_policy_id"]
        for row in sentinel_rows
        if isinstance(row.get("opponent_policy_id"), str) and row.get("opponent_policy_id")
    }
    missing = [opponent for opponent in config.sentinel_opponents if not _matches_any(opponent, present)]
    for opponent in missing:
        failures.append({"reason": "missing_sentinel_row", "opponent": opponent})
    for row in fixed_rows:
        delta = _int_value(row.get("delta_wins"))
        if delta < -int(config.max_sentinel_fixed_row_drop_wins):
            failures.append(
                {
                    "reason": "sentinel_fixed_row_drop",
                    "opponent": row.get("opponent_policy_id"),
                    "delta_wins": delta,
                    "threshold": -int(config.max_sentinel_fixed_row_drop_wins),
                }
            )
    for row in learned_rows:
        delta = _int_value(row.get("delta_wins"))
        if delta < int(config.max_sentinel_learned_row_drop_wins):
            failures.append(
                {
                    "reason": "sentinel_learned_row_drop",
                    "opponent": row.get("opponent_policy_id"),
                    "delta_wins": delta,
                    "threshold": int(config.max_sentinel_learned_row_drop_wins),
                }
            )
    learned_delta_wins = sum(_int_value(row.get("delta_wins")) for row in learned_rows)
    if learned_rows and learned_delta_wins < int(config.min_sentinel_learned_delta_wins):
        failures.append(
            {
                "reason": "sentinel_learned_aggregate_drop",
                "delta_wins": learned_delta_wins,
                "threshold": int(config.min_sentinel_learned_delta_wins),
            }
        )
    return {
        "passed": not failures,
        "failures": failures,
        "present_count": len(sentinel_rows),
        "expected_count": len(tuple(config.sentinel_opponents)),
        "fixed_delta_wins": sum(_int_value(row.get("delta_wins")) for row in fixed_rows),
        "learned_delta_wins": learned_delta_wins,
        "all_delta_wins": sum(_int_value(row.get("delta_wins")) for row in sentinel_rows),
    }


def _evaluate_full_gate(
    *,
    row_summaries: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Any],
    paired_seeds: int | None,
    config: MainLeagueFrontierScorecardConfig,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    fixed_delta_wins = _group_int(groups, "fixed_baselines", "delta_wins")
    learned_delta_wins = _group_int(groups, "learned_opponents", "delta_wins")
    if fixed_delta_wins is None:
        failures.append({"reason": "missing_fixed_group_delta"})
    elif fixed_delta_wins < int(config.min_full_fixed_delta_wins):
        failures.append(
            {
                "reason": "full_fixed_aggregate_drop",
                "delta_wins": fixed_delta_wins,
                "threshold": int(config.min_full_fixed_delta_wins),
            }
        )
    if learned_delta_wins is None:
        failures.append({"reason": "missing_learned_group_delta"})
    elif learned_delta_wins < int(config.min_full_learned_delta_wins):
        failures.append(
            {
                "reason": "full_learned_aggregate_drop",
                "delta_wins": learned_delta_wins,
                "threshold": int(config.min_full_learned_delta_wins),
            }
        )
    for row in row_summaries:
        if row.get("group") == "fixed":
            delta = _int_value(row.get("delta_wins"))
            if delta < -int(config.max_full_fixed_row_drop_wins):
                failures.append(
                    {
                        "reason": "full_fixed_row_drop",
                        "opponent": row.get("opponent_policy_id"),
                        "delta_wins": delta,
                        "threshold": -int(config.max_full_fixed_row_drop_wins),
                    }
                )
            continue
        delta = _int_value(row.get("delta_wins"))
        learned_drop_threshold = (
            int(config.max_confirm256_learned_row_drop_wins)
            if paired_seeds is not None and paired_seeds >= 256
            else int(config.max_confirm128_learned_row_drop_wins)
        )
        if delta < learned_drop_threshold:
            failures.append(
                {
                    "reason": "full_learned_row_drop",
                    "opponent": row.get("opponent_policy_id"),
                    "delta_wins": delta,
                    "threshold": learned_drop_threshold,
                }
            )
    return {
        "passed": not failures,
        "failures": failures,
        "fixed_delta_wins": fixed_delta_wins,
        "learned_delta_wins": learned_delta_wins,
        "all_delta_wins": _group_int(groups, "all_compared", "delta_wins"),
    }


def _recommend_escalation(
    *,
    panel_kind: str,
    paired_seeds: int | None,
    sentinel_gate: Mapping[str, Any],
    full_gate: Mapping[str, Any],
) -> dict[str, Any]:
    if panel_kind == "sentinel":
        if not sentinel_gate.get("passed"):
            return {"decision": "stop", "reason": "sentinel_gate_failed"}
        return {"decision": "run_full_confirm64", "reason": "sentinel_gate_passed"}
    if panel_kind != "full":
        return {"decision": "needs_panel", "reason": "compare_report_missing_required_rows"}
    if not full_gate.get("passed"):
        return {"decision": "stop", "reason": "full_gate_failed"}
    if paired_seeds is None:
        return {"decision": "needs_seed_count", "reason": "could_not_infer_paired_seeds"}
    if paired_seeds < 64:
        return {"decision": "run_full_confirm64", "reason": "full_gate_passed_below_confirm64"}
    if paired_seeds < 128:
        return {"decision": "run_confirm128", "reason": "full_confirm64_gate_passed"}
    if paired_seeds < 256:
        return {"decision": "run_confirm256", "reason": "confirm128_gate_passed"}
    return {"decision": "publishable_gate_candidate", "reason": "confirm256_gate_passed"}


def _classify_panel(
    *,
    row_summaries: Sequence[Mapping[str, Any]],
    fixed_opponents: Sequence[str],
    sentinel_opponents: Sequence[str],
) -> str:
    present_rows = [row for row in row_summaries if _row_is_resolved(row)]
    present = [str(row.get("opponent_policy_id") or "") for row in present_rows]
    if len(present_rows) == len(tuple(sentinel_opponents)) and all(
        _matches_any(opponent, present) for opponent in sentinel_opponents
    ):
        return "sentinel"
    fixed_count = sum(1 for row in present_rows if row.get("group") == "fixed")
    learned_count = sum(1 for row in present_rows if row.get("group") == "learned")
    if fixed_count >= len(tuple(fixed_opponents)) and learned_count > 0:
        return "full"
    return "partial"


def _row_is_resolved(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "ok" and isinstance(row.get("shared_games"), int)


def _infer_paired_seeds(row_summaries: Sequence[Mapping[str, Any]]) -> int | None:
    shared_games = [
        int(row["shared_games"])
        for row in row_summaries
        if isinstance(row.get("shared_games"), int) and int(row["shared_games"]) > 0
    ]
    if not shared_games:
        return None
    return min(shared_games) // 2


def _count_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(entries),
        "stop": 0,
        "run_full_confirm64": 0,
        "run_confirm128": 0,
        "run_confirm256": 0,
        "publishable_gate_candidate": 0,
    }
    for entry in entries:
        escalation = entry.get("escalation")
        decision = escalation.get("decision") if isinstance(escalation, Mapping) else None
        if isinstance(decision, str):
            counts[decision] = counts.get(decision, 0) + 1
    return counts


def _matches_any(value: str, candidates: Sequence[str] | set[str]) -> bool:
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


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
