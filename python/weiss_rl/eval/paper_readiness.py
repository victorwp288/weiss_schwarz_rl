"""Paper-readiness auditing over run directories and final-eval artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID
from weiss_rl.eval.readiness import contracts as _contracts
from weiss_rl.eval.readiness.guardrails import (
    build_final_eval_guardrail_summary as _build_final_eval_guardrail_summary,
)

DEFAULT_BASELINE_POLICY_ID = RANDOM_LEGAL_POLICY_ID
DEFAULT_BASELINE_POSTERIOR_MIN = 0.95
DEFAULT_BASELINE_WIN_RATE_THRESHOLD = 0.55
DEFAULT_SEAT_BIAS_MAX_ABS_DELTA = 0.05
DEFAULT_SEAT_BIAS_POSTERIOR_MIN = 0.95
DEFAULT_TRUNCATION_MAX_RATE = 0.02

_REQUIRED_SENSITIVITY_CASE_IDS = _contracts.REQUIRED_SENSITIVITY_CASE_IDS
RequiredArtifactSpec = _contracts.RequiredArtifactSpec
_build_final_eval_artifact_contract = _contracts.build_final_eval_artifact_contract
_build_manifest_contract = _contracts.build_manifest_contract
_build_run_directory_audit = _contracts.build_run_directory_audit
_evaluate_required_artifact = _contracts.evaluate_required_artifact
_required_run_artifact_specs = _contracts.required_run_artifact_specs
_resolve_sensitivity_summary_path = _contracts.resolve_sensitivity_summary_path
_sensitivity_root_candidates = _contracts.sensitivity_root_candidates
_validate_final_eval_policy_set = _contracts.validate_final_eval_policy_set
_validate_sensitivity_summary = _contracts.validate_sensitivity_summary

__all__ = [
    "DEFAULT_BASELINE_POLICY_ID",
    "DEFAULT_BASELINE_POSTERIOR_MIN",
    "DEFAULT_BASELINE_WIN_RATE_THRESHOLD",
    "DEFAULT_SEAT_BIAS_MAX_ABS_DELTA",
    "DEFAULT_SEAT_BIAS_POSTERIOR_MIN",
    "DEFAULT_TRUNCATION_MAX_RATE",
    "build_paper_readiness_summary",
    "write_paper_readiness_json",
]


def build_paper_readiness_summary(
    *,
    run_dir: Path | None = None,
    final_eval_dir: Path | None = None,
    focal_policy_id: str | None = None,
    baseline_policy_id: str = DEFAULT_BASELINE_POLICY_ID,
    max_truncation_rate: float = DEFAULT_TRUNCATION_MAX_RATE,
    seat_bias_max_abs_delta: float = DEFAULT_SEAT_BIAS_MAX_ABS_DELTA,
    seat_bias_posterior_min: float = DEFAULT_SEAT_BIAS_POSTERIOR_MIN,
    baseline_win_rate_threshold: float = DEFAULT_BASELINE_WIN_RATE_THRESHOLD,
    baseline_posterior_min: float = DEFAULT_BASELINE_POSTERIOR_MIN,
) -> dict[str, Any]:
    if (run_dir is None) == (final_eval_dir is None):
        raise ValueError("pass exactly one of run_dir or final_eval_dir")

    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
        layout = ArtifactLayout.from_run_dir(resolved_run_dir)
        return _build_run_directory_readiness_summary(
            run_dir=resolved_run_dir,
            final_eval_dir=layout.final_eval_dir,
            focal_policy_id=focal_policy_id,
            baseline_policy_id=baseline_policy_id,
            max_truncation_rate=max_truncation_rate,
            seat_bias_max_abs_delta=seat_bias_max_abs_delta,
            seat_bias_posterior_min=seat_bias_posterior_min,
            baseline_win_rate_threshold=baseline_win_rate_threshold,
            baseline_posterior_min=baseline_posterior_min,
        )

    resolved_final_eval_dir = Path(cast(Path, final_eval_dir))
    guardrails = _safe_build_final_eval_guardrail_summary(
        final_eval_dir=resolved_final_eval_dir,
        focal_policy_id=focal_policy_id,
        baseline_policy_id=baseline_policy_id,
        max_truncation_rate=max_truncation_rate,
        seat_bias_max_abs_delta=seat_bias_max_abs_delta,
        seat_bias_posterior_min=seat_bias_posterior_min,
        baseline_win_rate_threshold=baseline_win_rate_threshold,
        baseline_posterior_min=baseline_posterior_min,
    )
    if guardrails["loaded"]:
        payload = {
            "kind": "paper_readiness_summary_v2",
            "scope": "final_eval_dir",
            "passed": bool(guardrails["passed"]),
            "alarms": list(cast(Sequence[str], guardrails["alarms"])),
            "final_eval": dict(cast(Mapping[str, Any], guardrails["final_eval"])),
            "checks": dict(cast(Mapping[str, Any], guardrails["checks"])),
            "final_eval_guardrails": {
                "passed": bool(guardrails["passed"]),
                "alarms": list(cast(Sequence[str], guardrails["alarms"])),
                "message": "final_eval guardrails loaded successfully",
            },
        }
        return payload

    return {
        "kind": "paper_readiness_summary_v2",
        "scope": "final_eval_dir",
        "passed": False,
        "alarms": ["final_eval_guardrails"],
        "final_eval": dict(cast(Mapping[str, Any], guardrails["final_eval"])),
        "checks": {},
        "final_eval_guardrails": {
            "passed": False,
            "alarms": [],
            "reason": guardrails["reason"],
            "message": guardrails["message"],
        },
    }


def write_paper_readiness_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_run_directory_readiness_summary(
    *,
    run_dir: Path,
    final_eval_dir: Path,
    focal_policy_id: str | None,
    baseline_policy_id: str,
    max_truncation_rate: float,
    seat_bias_max_abs_delta: float,
    seat_bias_posterior_min: float,
    baseline_win_rate_threshold: float,
    baseline_posterior_min: float,
) -> dict[str, Any]:
    run_directory_audit = _build_run_directory_audit(run_dir)
    manifest_contract = _build_manifest_contract(run_dir)
    final_eval_artifact_contract = _build_final_eval_artifact_contract(final_eval_dir)
    guardrails = _safe_build_final_eval_guardrail_summary(
        final_eval_dir=final_eval_dir,
        focal_policy_id=focal_policy_id,
        baseline_policy_id=baseline_policy_id,
        max_truncation_rate=max_truncation_rate,
        seat_bias_max_abs_delta=seat_bias_max_abs_delta,
        seat_bias_posterior_min=seat_bias_posterior_min,
        baseline_win_rate_threshold=baseline_win_rate_threshold,
        baseline_posterior_min=baseline_posterior_min,
    )

    alarms: list[str] = []
    for section_name, section in (
        ("run_directory_audit", run_directory_audit),
        ("manifest_contract", manifest_contract),
        ("final_eval_artifact_contract", final_eval_artifact_contract),
    ):
        if not bool(section["passed"]):
            alarms.append(section_name)

    if guardrails["loaded"]:
        alarms.extend(str(alarm) for alarm in cast(Sequence[str], guardrails["alarms"]))
    else:
        alarms.append("final_eval_guardrails")

    return {
        "kind": "paper_readiness_summary_v2",
        "scope": "run_dir",
        "passed": not alarms,
        "alarms": alarms,
        "run_dir": {
            "dir": run_dir.as_posix(),
        },
        "final_eval": dict(cast(Mapping[str, Any], guardrails["final_eval"])),
        "checks": dict(cast(Mapping[str, Any], guardrails["checks"])),
        "run_directory_audit": run_directory_audit,
        "manifest_contract": manifest_contract,
        "final_eval_artifact_contract": final_eval_artifact_contract,
        "final_eval_guardrails": {
            "passed": bool(guardrails["passed"]),
            "alarms": list(cast(Sequence[str], guardrails["alarms"])),
            "reason": guardrails.get("reason"),
            "message": guardrails.get("message"),
        },
    }


def _safe_build_final_eval_guardrail_summary(
    *,
    final_eval_dir: Path,
    focal_policy_id: str | None,
    baseline_policy_id: str,
    max_truncation_rate: float,
    seat_bias_max_abs_delta: float,
    seat_bias_posterior_min: float,
    baseline_win_rate_threshold: float,
    baseline_posterior_min: float,
) -> dict[str, Any]:
    try:
        payload = _build_final_eval_guardrail_summary(
            final_eval_dir=final_eval_dir,
            focal_policy_id=focal_policy_id,
            baseline_policy_id=baseline_policy_id,
            max_truncation_rate=max_truncation_rate,
            seat_bias_max_abs_delta=seat_bias_max_abs_delta,
            seat_bias_posterior_min=seat_bias_posterior_min,
            baseline_win_rate_threshold=baseline_win_rate_threshold,
            baseline_posterior_min=baseline_posterior_min,
        )
    except Exception as exc:
        return {
            "loaded": False,
            "passed": False,
            "alarms": [],
            "final_eval": {
                "dir": final_eval_dir.as_posix(),
                "summary_path": (final_eval_dir / "summary.json").as_posix(),
                "policy_ids": [],
                "selection": {},
            },
            "checks": {},
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    return {
        "loaded": True,
        **payload,
        "reason": None,
        "message": None,
    }
