from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAST_LOOP_STAGE_DECISIONS = {
    "sentinel": None,
    "full_confirm64": "run_full_confirm64",
    "confirm128": "run_confirm128",
    "confirm256": "run_confirm256",
    "publish": "publishable_gate_candidate",
}


@dataclass(frozen=True, slots=True)
class MainLeagueFastLoopGateConfig:
    stage: str
    mechanistic_gate_json: Path | None = None
    target_gate_json: Path | None = None
    drift_gate_json: Path | None = None
    live_progress_gate_json: Path | None = None
    frontier_scorecard_json: Path | None = None
    candidate_label: str | None = None


def evaluate_main_league_fast_loop_gate(config: MainLeagueFastLoopGateConfig) -> dict[str, Any]:
    """Validate that a main-league probe is allowed to enter the requested eval stage."""

    stage = str(config.stage)
    if stage not in FAST_LOOP_STAGE_DECISIONS:
        raise ValueError(f"unknown fast-loop stage: {stage}")

    failures: list[dict[str, Any]] = []
    mechanistic_summary = _evaluate_mechanistic_gate(config.mechanistic_gate_json)
    if not mechanistic_summary["passed"]:
        failures.extend(mechanistic_summary["failures"])
    target_summary = _evaluate_target_gate(config.target_gate_json)
    if not target_summary["passed"]:
        failures.extend(target_summary["failures"])
    drift_summary = _evaluate_drift_gate(config.drift_gate_json)
    live_progress_summary = _evaluate_live_progress_gate(config.live_progress_gate_json)
    if stage == "sentinel":
        if config.drift_gate_json is None and config.live_progress_gate_json is None:
            failures.append({"reason": "missing_drift_or_live_progress_gate_json"})
    if not drift_summary["passed"]:
        failures.extend(drift_summary["failures"])
    if not live_progress_summary["passed"]:
        failures.extend(live_progress_summary["failures"])

    required_decision = FAST_LOOP_STAGE_DECISIONS[stage]
    scorecard_entry: Mapping[str, Any] | None = None
    if required_decision is not None:
        if config.frontier_scorecard_json is None:
            failures.append(
                {
                    "reason": "missing_frontier_scorecard_json",
                    "stage": stage,
                    "required_decision": required_decision,
                }
            )
        else:
            scorecard_payload = _read_json_object(config.frontier_scorecard_json)
            scorecard_entry = _select_scorecard_entry(scorecard_payload, candidate_label=config.candidate_label)
            if scorecard_entry is None:
                failures.append(
                    {
                        "reason": "scorecard_entry_not_found",
                        "candidate_label": config.candidate_label,
                        "scorecard_json": config.frontier_scorecard_json.as_posix(),
                    }
                )
            else:
                decision = _entry_decision(scorecard_entry)
                if decision != required_decision:
                    failures.append(
                        {
                            "reason": "wrong_escalation_decision",
                            "stage": stage,
                            "decision": decision,
                            "required_decision": required_decision,
                            "candidate_label": scorecard_entry.get("candidate_label"),
                        }
                    )

    return {
        "kind": "main_league_fast_loop_gate_v1",
        "passed": not failures,
        "stage": stage,
        "required_decision": required_decision,
        "failures": failures,
        "mechanistic_gate": mechanistic_summary,
        "target_gate": target_summary,
        "drift_gate": drift_summary,
        "live_progress_gate": live_progress_summary,
        "frontier_scorecard_json": None
        if config.frontier_scorecard_json is None
        else config.frontier_scorecard_json.as_posix(),
        "candidate_label": config.candidate_label,
        "scorecard_entry": _compact_scorecard_entry(scorecard_entry),
    }


def write_main_league_fast_loop_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_main_league_fast_loop_gate(config: MainLeagueFastLoopGateConfig) -> dict[str, Any]:
    report = evaluate_main_league_fast_loop_gate(config)
    if not bool(report.get("passed")):
        failures = report.get("failures")
        raise RuntimeError(f"main-league fast-loop gate failed for {config.stage}: {failures}")
    return report


def _evaluate_mechanistic_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "passed": False,
            "path": None,
            "failures": [{"reason": "missing_mechanistic_gate_json"}],
            "summary": {},
        }
    payload = _read_json_object(path)
    passed = bool(payload.get("passed"))
    failures = []
    if not passed:
        failures.append(
            {
                "reason": "mechanistic_gate_failed",
                "mechanistic_failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
            }
        )
    return {
        "passed": passed,
        "path": path.as_posix(),
        "failures": failures,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {},
    }


def _evaluate_drift_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "passed": True,
            "path": None,
            "failures": [],
            "summary": {},
        }
    payload = _read_json_object(path)
    passed = bool(payload.get("passed"))
    failures = []
    if not passed:
        failures.append(
            {
                "reason": "drift_gate_failed",
                "drift_failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
            }
        )
    return {
        "passed": passed,
        "path": path.as_posix(),
        "failures": failures,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {},
    }


def _evaluate_live_progress_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "passed": True,
            "path": None,
            "failures": [],
            "summary": {},
        }
    payload = _read_json_object(path)
    passed = bool(payload.get("passed"))
    failures = []
    if not passed:
        failures.append(
            {
                "reason": "live_progress_gate_failed",
                "live_progress_failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
            }
        )
    return {
        "passed": passed,
        "path": path.as_posix(),
        "failures": failures,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {},
    }


def _evaluate_target_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "passed": True,
            "path": None,
            "failures": [],
            "summary": {},
        }
    payload = _read_json_object(path)
    passed = bool(payload.get("passed"))
    failures = []
    if not passed:
        failures.append(
            {
                "reason": "target_gate_failed",
                "target_failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
            }
        )
    return {
        "passed": passed,
        "path": path.as_posix(),
        "failures": failures,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {},
    }


def _select_scorecard_entry(payload: Mapping[str, Any], *, candidate_label: str | None) -> Mapping[str, Any] | None:
    entries = [entry for entry in payload.get("entries", []) if isinstance(entry, Mapping)]
    if candidate_label:
        matches = [entry for entry in entries if str(entry.get("candidate_label") or "") == candidate_label]
        return matches[0] if len(matches) == 1 else None
    return entries[0] if len(entries) == 1 else None


def _entry_decision(entry: Mapping[str, Any] | None) -> str | None:
    if entry is None:
        return None
    escalation = entry.get("escalation")
    if not isinstance(escalation, Mapping):
        return None
    decision = escalation.get("decision")
    return str(decision) if isinstance(decision, str) else None


def _compact_scorecard_entry(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "candidate_label": entry.get("candidate_label"),
        "panel_kind": entry.get("panel_kind"),
        "paired_seeds": entry.get("paired_seeds"),
        "escalation": entry.get("escalation"),
        "sentinel_gate": entry.get("sentinel_gate"),
        "full_gate": entry.get("full_gate"),
        "group_deltas": entry.get("group_deltas"),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


__all__ = [
    "FAST_LOOP_STAGE_DECISIONS",
    "MainLeagueFastLoopGateConfig",
    "assert_main_league_fast_loop_gate",
    "evaluate_main_league_fast_loop_gate",
    "write_main_league_fast_loop_gate",
]
