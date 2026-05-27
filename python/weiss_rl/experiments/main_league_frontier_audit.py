"""Aggregate main-league successor evidence into a thesis-facing audit report."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MainLeagueFrontierAuditConfig:
    diagnostics_dir: Path
    output_json: Path | None = None
    date_token: str = "20260521"
    selected_run: str = "runs/main_champion_hardneg_interp_u10_repair_a015_20260517"
    selected_policy_id: str = "main_interp_repair_a015"
    max_entries: int = 500


def build_main_league_frontier_audit(config: MainLeagueFrontierAuditConfig) -> dict[str, Any]:
    diagnostics_dir = Path(config.diagnostics_dir)
    json_paths = sorted(diagnostics_dir.glob(f"*{config.date_token}*.json"))
    artifacts: list[dict[str, Any]] = []
    for path in json_paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        kind = _artifact_kind(payload, path)
        if kind == "ignored":
            continue
        artifacts.append(_artifact_summary(path, payload, kind=kind))

    scorecard_entries = _scorecard_entries(artifacts)
    compare_entries = [artifact for artifact in artifacts if artifact["kind"] == "paired_outcome_compare"]
    gate_entries = [artifact for artifact in artifacts if artifact["kind"].endswith("_gate")]
    candidate_records = _candidate_records(scorecard_entries, compare_entries, gate_entries)
    publishable = [record for record in candidate_records if bool(record["publishable_successor_candidate"])]
    confirm128_ready = [record for record in candidate_records if record["recommended_next_stage"] == "confirm128"]
    full64_ready = [record for record in candidate_records if record["recommended_next_stage"] == "full_confirm64"]
    stopped = [record for record in candidate_records if record["recommended_next_stage"] == "stop"]

    report = {
        "kind": "main_league_frontier_audit_v1",
        "date_token": str(config.date_token),
        "diagnostics_dir": diagnostics_dir.as_posix(),
        "selected_run": str(config.selected_run),
        "selected_policy_id": str(config.selected_policy_id),
        "artifact_count": len(artifacts),
        "scorecard_entry_count": len(scorecard_entries),
        "compare_entry_count": len(compare_entries),
        "gate_entry_count": len(gate_entries),
        "candidate_count": len(candidate_records),
        "decision": {
            "publishable_successor_exists": bool(publishable),
            "selected_remains_locked": not bool(publishable),
            "reason": "no_confirm256_publishable_successor" if not publishable else "published_successor_found",
            "confirm128_ready_count": len(confirm128_ready),
            "full_confirm64_ready_count": len(full64_ready),
            "stopped_candidate_count": len(stopped),
        },
        "counts": {
            "artifacts_by_kind": dict(sorted(Counter(str(artifact["kind"]) for artifact in artifacts).items())),
            "candidates_by_next_stage": dict(
                sorted(Counter(str(record["recommended_next_stage"]) for record in candidate_records).items())
            ),
            "scorecard_decisions": dict(sorted(Counter(str(entry["decision"]) for entry in scorecard_entries).items())),
        },
        "best_non_publishable_signals": _best_non_publishable(candidate_records),
        "candidate_records": candidate_records[: max(0, int(config.max_entries))],
        "artifacts": artifacts[: max(0, int(config.max_entries))],
    }
    return report


def write_main_league_frontier_audit(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_main_league_frontier_audit_markdown(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = report.get("decision") if isinstance(report.get("decision"), Mapping) else {}
    best = (
        report.get("best_non_publishable_signals")
        if isinstance(report.get("best_non_publishable_signals"), list)
        else []
    )
    next_step = [
        item
        for item in report.get("candidate_records", [])
        if isinstance(item, Mapping) and str(item.get("recommended_next_stage") or "") != "stop"
    ]
    lines = [
        "# Main League Frontier Audit",
        "",
        f"- Selected run: `{report.get('selected_run')}`",
        f"- Selected policy id: `{report.get('selected_policy_id')}`",
        f"- Publishable successor exists: `{str(bool(decision.get('publishable_successor_exists'))).lower()}`",
        f"- Decision: `{decision.get('reason')}`",
        f"- Candidate records: `{report.get('candidate_count')}`",
        f"- Scorecard entries: `{report.get('scorecard_entry_count')}`",
        f"- Gate artifacts: `{report.get('gate_entry_count')}`",
        "",
        "## Best Non-Publishable Signals",
        "",
    ]
    if best:
        lines.append("| candidate | panel | seeds | fixed delta | learned delta | decision | reason |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for item in best[:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(str(item.get("candidate_label") or "")),
                        _md(str(item.get("panel_kind") or "")),
                        _md(str(item.get("paired_seeds") or "")),
                        _md(str(item.get("fixed_delta_wins") or 0)),
                        _md(str(item.get("learned_delta_wins") or 0)),
                        _md(str(item.get("decision") or "")),
                        _md(str(item.get("reason") or "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No scorecard entries were found.")
    lines.extend(["", "## Remaining Non-Stop Records", ""])
    if next_step:
        lines.append("| candidate | stage | scorecards | gates | compares | note |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in next_step[:20]:
            stage = str(item.get("recommended_next_stage") or "")
            note = "mechanistic or gate-only survivor" if stage == "sentinel" else "compare evidence without scorecard"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md(str(item.get("candidate_label") or "")),
                        _md(stage),
                        _md(str(item.get("scorecard_count") or 0)),
                        _md(str(item.get("gate_count") or 0)),
                        _md(str(item.get("compare_count") or 0)),
                        _md(note),
                    ]
                )
                + " |"
            )
    else:
        lines.append("All candidate families are stopped or published.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "No model should replace the locked selected checkpoint unless the JSON audit reports a publishable successor. "
            "At this snapshot, the audit keeps selected locked unless a successor has actual publish-level evidence. "
            "Mechanistic-only and sentinel-only survivors are next-step candidates, not selected-model evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_kind(payload: Mapping[str, Any], path: Path) -> str:
    kind = str(payload.get("kind") or "")
    name = path.name
    if kind == "main_league_fast_loop_gate_v1":
        return "fast_loop_gate"
    if kind == "paired_targeted_outcome_compare_v1":
        return "paired_outcome_compare"
    if kind == "paired_outcome_compare_gate_v1":
        return "paired_outcome_compare_gate"
    if kind == "paired_outcome_preference_row_guard_v1":
        return "row_guard_gate"
    if kind == "paired_outcome_preference_edge_margin_report_v1":
        return "edge_margin_gate"
    if kind == "paired_outcome_preference_mechanistic_gate_v1":
        return "preference_mechanistic_gate"
    if kind == "trajectory_policy_drift_gate_v1":
        return "trajectory_drift_gate"
    if kind == "main_league_live_probe_gate_v1":
        return "live_probe_gate"
    if "main_league_frontier_scorecard" in name and "entries" in payload:
        return "frontier_scorecard"
    return "ignored"


def _artifact_summary(path: Path, payload: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "frontier_scorecard":
        return _frontier_scorecard_summary(path, payload)
    if kind == "paired_outcome_compare":
        return _paired_compare_summary(path, payload)
    if kind.endswith("_gate"):
        return _gate_summary(path, payload, kind=kind)
    return {"kind": kind, "path": path.as_posix()}


def _frontier_scorecard_summary(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    return {
        "kind": "frontier_scorecard",
        "path": path.as_posix(),
        "entry_count": len(entries),
        "counts": payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {},
        "entries": [_scorecard_entry_summary(path, entry) for entry in entries if isinstance(entry, Mapping)],
    }


def _scorecard_entry_summary(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    escalation = entry.get("escalation") if isinstance(entry.get("escalation"), Mapping) else {}
    full_gate = entry.get("full_gate") if isinstance(entry.get("full_gate"), Mapping) else {}
    sentinel_gate = entry.get("sentinel_gate") if isinstance(entry.get("sentinel_gate"), Mapping) else {}
    group_deltas = entry.get("group_deltas") if isinstance(entry.get("group_deltas"), Mapping) else {}
    decision = str(escalation.get("decision") or "unknown")
    reason = str(escalation.get("reason") or "")
    return {
        "source_scorecard_json": path.as_posix(),
        "candidate_label": str(entry.get("candidate_label") or ""),
        "baseline_label": str(entry.get("baseline_label") or ""),
        "panel_kind": str(entry.get("panel_kind") or ""),
        "paired_seeds": _int_or_none(entry.get("paired_seeds")),
        "decision": decision,
        "reason": reason,
        "publishable_successor_candidate": decision == "publish_candidate",
        "recommended_next_stage": _next_stage_from_decision(decision),
        "fixed_delta_wins": _int_or_zero(group_deltas.get("fixed_delta_wins")),
        "learned_delta_wins": _int_or_zero(group_deltas.get("learned_delta_wins")),
        "all_delta_wins": _int_or_zero(group_deltas.get("all_delta_wins")),
        "fixed_changed_outcome": _int_or_zero(group_deltas.get("fixed_changed_outcome")),
        "learned_changed_outcome": _int_or_zero(group_deltas.get("learned_changed_outcome")),
        "full_gate_passed": _bool_or_none(full_gate.get("passed")),
        "sentinel_gate_passed": _bool_or_none(sentinel_gate.get("passed")),
        "gate_failures": _gate_failures(full_gate) + _gate_failures(sentinel_gate),
        "row_regressions": entry.get("row_regressions") if isinstance(entry.get("row_regressions"), Mapping) else {},
    }


def _paired_compare_summary(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), Mapping) else {}
    groups = payload.get("groups") if isinstance(payload.get("groups"), Mapping) else {}
    all_compared = groups.get("all_compared") if isinstance(groups.get("all_compared"), Mapping) else {}
    fixed = groups.get("fixed_baselines") if isinstance(groups.get("fixed_baselines"), Mapping) else {}
    learned = groups.get("learned_opponents") if isinstance(groups.get("learned_opponents"), Mapping) else {}
    return {
        "kind": "paired_outcome_compare",
        "path": path.as_posix(),
        "candidate_label": str(candidate.get("label") or ""),
        "baseline_label": str(baseline.get("label") or ""),
        "all_delta_wins": _int_or_zero(all_compared.get("delta_wins")),
        "fixed_delta_wins": _int_or_zero(fixed.get("delta_wins")),
        "learned_delta_wins": _int_or_zero(learned.get("delta_wins")),
        "shared_games": _int_or_zero(all_compared.get("shared_games")),
        "fixed_shared_games": _int_or_zero(fixed.get("shared_games")),
        "learned_shared_games": _int_or_zero(learned.get("shared_games")),
    }


def _gate_summary(path: Path, payload: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    candidate_label = str(payload.get("candidate_label") or "")
    if not candidate_label and isinstance(payload.get("failures"), list):
        for failure in payload.get("failures", []):
            if isinstance(failure, Mapping) and failure.get("candidate_label"):
                candidate_label = str(failure.get("candidate_label"))
                break
    return {
        "kind": kind,
        "path": path.as_posix(),
        "candidate_label": candidate_label,
        "passed": bool(payload.get("passed")) if "passed" in payload else not bool(payload.get("failures")),
        "failures": _failure_strings(payload.get("failures")),
        "summary": dict(summary),
    }


def _scorecard_entries(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.get("kind") != "frontier_scorecard":
            continue
        for entry in artifact.get("entries", []):
            if isinstance(entry, Mapping):
                entries.append(dict(entry))
    return entries


def _candidate_records(
    scorecard_entries: Sequence[Mapping[str, Any]],
    compare_entries: Sequence[Mapping[str, Any]],
    gate_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for entry in scorecard_entries:
        label = _canonical_candidate_label(str(entry.get("candidate_label") or ""))
        if not label:
            continue
        record = by_label.setdefault(label, _empty_candidate_record(label))
        record["evidence_labels"].add(str(entry.get("candidate_label") or ""))
        record["scorecards"].append(dict(entry))
    for entry in compare_entries:
        label = _canonical_candidate_label(str(entry.get("candidate_label") or ""))
        if not label:
            continue
        record = by_label.setdefault(label, _empty_candidate_record(label))
        record["evidence_labels"].add(str(entry.get("candidate_label") or ""))
        record["compares"].append(dict(entry))
    for entry in gate_entries:
        label = _canonical_candidate_label(str(entry.get("candidate_label") or ""))
        if not label:
            label = _canonical_candidate_label(_label_from_filename(str(entry.get("path") or "")))
        record = by_label.setdefault(label, _empty_candidate_record(label))
        record["evidence_labels"].add(
            str(entry.get("candidate_label") or "") or _label_from_filename(str(entry.get("path") or ""))
        )
        record["gates"].append(dict(entry))

    records = []
    for label, record in sorted(by_label.items()):
        scorecards = record["scorecards"]
        gates = record["gates"]
        compares = record["compares"]
        recommended = _recommended_stage(scorecards, gates, compares)
        publishable = any(bool(item.get("publishable_successor_candidate")) for item in scorecards)
        best_scorecard = _best_scorecard(scorecards)
        records.append(
            {
                "candidate_label": label,
                "evidence_labels": sorted(str(item) for item in record["evidence_labels"] if str(item)),
                "recommended_next_stage": recommended,
                "publishable_successor_candidate": publishable,
                "scorecard_count": len(scorecards),
                "compare_count": len(compares),
                "gate_count": len(gates),
                "passed_gate_count": sum(1 for gate in gates if bool(gate.get("passed"))),
                "failed_gate_count": sum(1 for gate in gates if not bool(gate.get("passed"))),
                "best_scorecard": best_scorecard,
                "gate_failures": [
                    {
                        "kind": gate.get("kind"),
                        "path": gate.get("path"),
                        "failures": gate.get("failures"),
                    }
                    for gate in gates
                    if not bool(gate.get("passed"))
                ],
                "compares": compares,
            }
        )
    return records


def _empty_candidate_record(label: str) -> dict[str, Any]:
    return {"candidate_label": label, "evidence_labels": set(), "scorecards": [], "compares": [], "gates": []}


def _recommended_stage(
    scorecards: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    compares: Sequence[Mapping[str, Any]],
) -> str:
    if any(str(item.get("decision") or "") == "publish_candidate" for item in scorecards):
        return "publish"
    decisions = {str(item.get("decision") or "") for item in scorecards}
    if "stop" in decisions:
        return "stop"
    if gates and any(not bool(gate.get("passed")) for gate in gates):
        return "stop"
    if "run_confirm256" in decisions:
        return "confirm256"
    if "run_confirm128" in decisions:
        return "confirm128"
    if "run_full_confirm64" in decisions:
        return "full_confirm64"
    if _compare_evidence_stops(compares):
        return "stop"
    if gates and all(bool(gate.get("passed")) for gate in gates):
        return "sentinel"
    if compares:
        return "evidence_only"
    return "unknown"


def _best_scorecard(scorecards: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not scorecards:
        return None
    return max(
        (dict(item) for item in scorecards),
        key=lambda item: (
            _int_or_zero(item.get("paired_seeds")),
            _int_or_zero(item.get("learned_delta_wins")),
            _int_or_zero(item.get("fixed_delta_wins")),
            _int_or_zero(item.get("all_delta_wins")),
        ),
    )


def _compare_evidence_stops(compares: Sequence[Mapping[str, Any]]) -> bool:
    """Treat already-run sentinel/full compares with no learned movement as stopped evidence."""

    for compare in compares:
        shared_games = _int_or_zero(compare.get("shared_games"))
        fixed_delta = _int_or_zero(compare.get("fixed_delta_wins"))
        learned_delta = _int_or_zero(compare.get("learned_delta_wins"))
        if shared_games >= 224 and (fixed_delta < 0 or learned_delta <= 0):
            return True
    return False


def _best_non_publishable(candidate_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    non_publishable: list[dict[str, Any]] = []
    for record in candidate_records:
        if bool(record.get("publishable_successor_candidate")):
            continue
        scorecard = record.get("best_scorecard")
        if not isinstance(scorecard, Mapping):
            continue
        item = dict(scorecard)
        item["candidate_label"] = str(record.get("candidate_label") or scorecard.get("candidate_label") or "")
        item["recommended_next_stage"] = str(record.get("recommended_next_stage") or "")
        non_publishable.append(item)
    ranked = sorted(
        non_publishable,
        key=lambda item: (
            _int_or_zero(item.get("learned_delta_wins")),
            _int_or_zero(item.get("fixed_delta_wins")),
            _int_or_zero(item.get("all_delta_wins")),
            _int_or_zero(item.get("paired_seeds")),
        ),
        reverse=True,
    )
    return [dict(item) for item in ranked[:20]]


def _next_stage_from_decision(decision: str) -> str:
    if decision == "publish_candidate":
        return "publish"
    if decision == "run_confirm256":
        return "confirm256"
    if decision == "run_confirm128":
        return "confirm128"
    if decision == "run_full_confirm64":
        return "full_confirm64"
    if decision == "stop":
        return "stop"
    return "unknown"


def _gate_failures(gate: Mapping[str, Any]) -> list[str]:
    failures = gate.get("failures")
    if not isinstance(failures, list):
        return []
    return _failure_strings(failures)


def _failure_strings(failures: object) -> list[str]:
    if not isinstance(failures, list):
        return []
    result: list[str] = []
    for item in failures:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            reason = str(item.get("reason") or item)
            result.append(reason)
        else:
            result.append(str(item))
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _label_from_filename(path_text: str) -> str:
    stem = Path(path_text).stem
    for prefix in (
        "main_league_fast_loop_gate_",
        "main_league_frontier_scorecard_",
        "paired_outcome_compare_gate_",
        "paired_outcome_compare_",
        "paired_outcome_preference_row_guard_",
        "paired_outcome_preference_gate_",
        "paired_outcome_preference_edge_gate_",
        "paired_outcome_preference_edge_report_",
        "paired_outcome_preference_edge_margins_",
        "paired_outcome_preference_mechanistic_gate_",
        "trajectory_policy_drift_gate_",
        "main_league_live_probe_gate_",
    ):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


_DATE_SUFFIX_RE = re.compile(r"_(?:20\d{6})(?:_\d+)?$")


def _canonical_candidate_label(label: str) -> str:
    """Collapse stage/depth labels so a candidate's lifecycle is audited together."""

    result = str(label or "").strip()
    if not result:
        return ""
    result = _apply_label_aliases(result)
    changed = True
    while changed:
        changed = False
        before = result
        result = _DATE_SUFFIX_RE.sub("", result)
        result = _strip_known_suffixes(result)
        result = _strip_known_prefixes(result)
        result = _apply_label_aliases(result)
        changed = result != before
    return result


def _strip_known_suffixes(label: str) -> str:
    suffixes = (
        "_vs_selected",
        "_vs_selected_fullsurface",
        "_vs_selected_samepublic_surface",
        "_same_seed_surface",
        "_selected_parent_children",
        "_on_fullsurface",
        "_fullsurface",
        "_samepublic_surface",
        "_improvement",
        "_full_confirm64_blocked",
        "_confirm128_blocked",
        "_confirm256_blocked",
        "_full13_confirm64",
        "_full_confirm64",
        "_confirm256",
        "_confirm128",
        "_confirm64",
        "_sentinel16_triage",
        "_sentinel16",
        "_sentinel",
        "_labeled",
        "_strict_labeled",
        "_strict",
    )
    for suffix in suffixes:
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _strip_known_prefixes(label: str) -> str:
    prefixes = (
        "exactrepair4_",
        "sentinel16_",
        "neartie_",
    )
    for prefix in prefixes:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _apply_label_aliases(label: str) -> str:
    replacements = (
        ("a050_spanfilter_actionlabel_pref_", "spanpref_"),
        ("a050_context_rich_spanfilter_refret_", "refret_"),
        ("a050p2_parent_child_interp_", "interp_"),
        ("a050p2_unlocked_rowdeficit_", "a050p2_live_unlocked_rowdeficit_"),
    )
    for prefix, replacement in replacements:
        if label.startswith(prefix):
            return replacement + label[len(prefix) :]
    if label.startswith("a075_balanced_contextpref_conflictfiltered_pairw9x2_fullbatch_lrscale0001_e2"):
        return "pairw9x2_e2"
    if label.startswith("a075_balanced_contextpref_conflictfiltered_pairw9x2_fullbatch_lrscale0001_u1"):
        return "pairw9x2_lr0001"
    if label == "refret_lr001_u1":
        return "refret_lr001"
    if label == "refret_lr0001_u1":
        return "refret_lr0001"
    if label == "unlocked_rowdeficit_u1":
        return "a050p2_live_unlocked_rowdeficit_u1"
    return label


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _md(value: str) -> str:
    return value.replace("|", "\\|")


__all__ = [
    "MainLeagueFrontierAuditConfig",
    "build_main_league_frontier_audit",
    "write_main_league_frontier_audit",
    "write_main_league_frontier_audit_markdown",
]
