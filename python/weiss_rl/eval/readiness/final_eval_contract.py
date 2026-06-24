"""Final-eval artifact integrity checks for paper-readiness audits."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.eval.readiness.fields import (
    load_json_object as _load_json_object,
)
from weiss_rl.eval.readiness.final_eval_matchup_contract import (
    build_final_eval_matchup_contract as _build_final_eval_matchup_contract,
)
from weiss_rl.eval.readiness.final_eval_summary import (
    policy_ids as _policy_ids,
)
from weiss_rl.eval.readiness.final_eval_summary import (
    summary_section_keys as _summary_section_keys,
)
from weiss_rl.eval.readiness.sensitivity_contract import (
    resolve_sensitivity_summary_path,
    sensitivity_root_candidates,
    validate_sensitivity_summary,
)


def build_final_eval_artifact_contract(final_eval_dir: Path) -> dict[str, Any]:
    summary_path = final_eval_dir / "summary.json"
    try:
        payload = _load_json_object(summary_path)
    except Exception as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": [],
            "expected_matchup_count": None,
            "observed_matchup_count": None,
            "missing_matchups": [],
            "duplicate_matchups": [],
            "noncanonical_matchups": [],
            "reference_failures": [],
            "sensitivity_cases": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    policy_ids = _policy_ids(payload)
    summary_sections = _summary_section_keys(payload)
    if len(set(policy_ids)) != len(policy_ids):
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "summary_section_keys": summary_sections,
            "expected_matchup_count": None,
            "observed_matchup_count": None,
            "missing_matchups": [],
            "duplicate_matchups": [],
            "noncanonical_matchups": [],
            "reference_failures": [],
            "sensitivity_cases": [],
            "reason": "duplicate_policy_ids",
            "message": "final_eval summary policy_ids must be unique",
        }

    policy_set_check = validate_final_eval_policy_set(final_eval_dir=final_eval_dir, policy_ids=policy_ids)
    sensitivity_check = validate_sensitivity_summary(final_eval_dir=final_eval_dir, policy_ids=policy_ids)
    matchup_contract = _build_final_eval_matchup_contract(
        final_eval_dir=final_eval_dir,
        summary_payload=payload,
        policy_ids=policy_ids,
    )

    if matchup_contract.get("reason") == "invalid_matchup_index":
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "summary_section_keys": summary_sections,
            "expected_matchup_count": matchup_contract["expected_matchup_count"],
            "observed_matchup_count": matchup_contract["observed_matchup_count"],
            "missing_matchups": matchup_contract["missing_matchups"],
            "duplicate_matchups": matchup_contract["duplicate_matchups"],
            "noncanonical_matchups": matchup_contract["noncanonical_matchups"],
            "reference_failures": matchup_contract["reference_failures"],
            "policy_set": policy_set_check,
            "sensitivity_summary": sensitivity_check,
            "reason": "invalid_matchup_index",
            "message": matchup_contract["message"],
        }

    passed = bool(matchup_contract["passed"]) and bool(policy_set_check["passed"]) and bool(sensitivity_check["passed"])

    return {
        "passed": passed,
        "summary_path": summary_path.as_posix(),
        "policy_ids": list(policy_ids),
        "summary_section_keys": summary_sections,
        "expected_matchup_count": matchup_contract["expected_matchup_count"],
        "observed_matchup_count": matchup_contract["observed_matchup_count"],
        "missing_matchups": matchup_contract["missing_matchups"],
        "duplicate_matchups": matchup_contract["duplicate_matchups"],
        "noncanonical_matchups": matchup_contract["noncanonical_matchups"],
        "reference_failures": matchup_contract["reference_failures"],
        "policy_set": policy_set_check,
        "sensitivity_summary": sensitivity_check,
        "message": (
            "final_eval artifact contract is complete"
            if passed
            else "final_eval artifact contract is missing required referenced artifacts"
        ),
    }


def validate_final_eval_policy_set(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
    policy_set_path = final_eval_dir / "policy_set.json"
    try:
        payload = _load_json_object(policy_set_path)
    except Exception as exc:
        return {
            "passed": False,
            "policy_set_path": policy_set_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    recorded_policy_ids = payload.get("policy_ids")
    if not isinstance(recorded_policy_ids, list) or any(not isinstance(item, str) for item in recorded_policy_ids):
        return {
            "passed": False,
            "policy_set_path": policy_set_path.as_posix(),
            "reason": "invalid_policy_ids",
            "message": "final_eval policy_set.json must include string policy_ids",
        }
    return {
        "passed": list(recorded_policy_ids) == list(policy_ids),
        "policy_set_path": policy_set_path.as_posix(),
        "policy_ids": list(recorded_policy_ids),
        "message": (
            "policy_set.json matches summary policy_ids"
            if list(recorded_policy_ids) == list(policy_ids)
            else "policy_set.json policy_ids do not match summary policy_ids"
        ),
    }


__all__ = [
    "build_final_eval_artifact_contract",
    "resolve_sensitivity_summary_path",
    "sensitivity_root_candidates",
    "validate_final_eval_policy_set",
    "validate_sensitivity_summary",
]
