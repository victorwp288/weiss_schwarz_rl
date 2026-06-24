"""Sensitivity artifact checks for paper-readiness audits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.eval.readiness.fields import load_json_object as _load_json_object
from weiss_rl.eval.readiness.specs import REQUIRED_SENSITIVITY_CASE_IDS


def sensitivity_root_candidates(*, final_eval_dir: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    canonical = final_eval_dir.parent / "metagame"
    legacy = final_eval_dir / "sensitivity"
    for candidate in (canonical, legacy):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def resolve_sensitivity_summary_path(final_eval_dir: Path) -> Path:
    for candidate_root in sensitivity_root_candidates(final_eval_dir=final_eval_dir):
        summary_path = candidate_root / "summary.json"
        if summary_path.is_file():
            return summary_path
    return sensitivity_root_candidates(final_eval_dir=final_eval_dir)[0] / "summary.json"


def validate_sensitivity_summary(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
    summary_path = resolve_sensitivity_summary_path(final_eval_dir)
    try:
        payload = _load_json_object(summary_path)
    except Exception as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "cases": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, dict):
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "cases": [],
            "reason": "missing_cases",
            "message": "sensitivity summary must include a cases object",
        }
    cases = sorted(str(case_id) for case_id in raw_cases)
    missing_cases = [case_id for case_id in REQUIRED_SENSITIVITY_CASE_IDS if case_id not in raw_cases]
    payload_policy_ids = payload.get("policy_ids")
    policy_ids_match = isinstance(payload_policy_ids, list) and payload_policy_ids == list(policy_ids)
    passed = not missing_cases and policy_ids_match
    return {
        "passed": passed,
        "summary_path": summary_path.as_posix(),
        "cases": cases,
        "missing_cases": missing_cases,
        "policy_ids_match": policy_ids_match,
        "message": (
            "sensitivity summary covers S0-S2 and matches final_eval policy_ids"
            if passed
            else "sensitivity summary is missing required cases or mismatches final_eval policy_ids"
        ),
    }


__all__ = [
    "resolve_sensitivity_summary_path",
    "sensitivity_root_candidates",
    "validate_sensitivity_summary",
]
