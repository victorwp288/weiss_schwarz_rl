"""Paper-readiness auditing over run directories and final-eval artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from scipy.stats import beta as beta_dist

from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
)

DEFAULT_BASELINE_POLICY_ID = RANDOM_LEGAL_POLICY_ID
DEFAULT_BASELINE_POSTERIOR_MIN = 0.95
DEFAULT_BASELINE_WIN_RATE_THRESHOLD = 0.55
DEFAULT_SEAT_BIAS_MAX_ABS_DELTA = 0.05
DEFAULT_SEAT_BIAS_POSTERIOR_MIN = 0.95
DEFAULT_TRUNCATION_MAX_RATE = 0.02
_REQUIRED_SENSITIVITY_CASE_IDS = ("S0", "S1", "S2")

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


@dataclass(frozen=True, slots=True)
class RequiredArtifactSpec:
    artifact_id: str
    description: str
    category: str
    paths: tuple[Path, ...] = ()
    glob: str | None = None
    minimum_count: int = 1


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
        resolved_final_eval_dir = resolved_run_dir / "eval" / "final_eval"
        return _build_run_directory_readiness_summary(
            run_dir=resolved_run_dir,
            final_eval_dir=resolved_final_eval_dir,
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


def _build_run_directory_audit(run_dir: Path) -> dict[str, Any]:
    artifact_results = {
        spec.artifact_id: _evaluate_required_artifact(run_dir=run_dir, spec=spec)
        for spec in _required_run_artifact_specs()
    }
    missing_artifacts = [artifact_id for artifact_id, result in artifact_results.items() if not bool(result["passed"])]
    return {
        "passed": not missing_artifacts,
        "artifact_count": len(artifact_results),
        "missing_artifacts": missing_artifacts,
        "artifacts": artifact_results,
        "message": (
            "all required run-directory artifacts are present"
            if not missing_artifacts
            else f"missing {len(missing_artifacts)} required artifact checks"
        ),
    }


def _required_run_artifact_specs() -> tuple[RequiredArtifactSpec, ...]:
    specs = [
        RequiredArtifactSpec(
            artifact_id="run_manifest",
            description="Run manifest JSON",
            category="run_root",
            paths=(Path("manifest.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="spec_bundle_json",
            description="Spec bundle JSON",
            category="run_root",
            paths=(Path("spec_bundle.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="spec_hash_txt",
            description="Spec bundle SHA-256 text file",
            category="run_root",
            paths=(Path("spec_hash256.txt"),),
        ),
        RequiredArtifactSpec(
            artifact_id="config_hash_txt",
            description="Config SHA-256 text file",
            category="run_root",
            paths=(Path("config_hash256.txt"),),
        ),
        RequiredArtifactSpec(
            artifact_id="config_canonical_json",
            description="Canonical config JSON",
            category="run_root",
            paths=(Path("config_canonical.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="training_metrics",
            description="Training metrics JSONL",
            category="training",
            paths=(Path("training/logs/training_metrics.jsonl"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_summary",
            description="Final-eval summary JSON",
            category="final_eval",
            paths=(Path("eval/final_eval/summary.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_policy_set",
            description="Final-eval policy set JSON",
            category="final_eval",
            paths=(Path("eval/final_eval/policy_set.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_posterior_samples",
            description="Final-eval posterior samples JSON",
            category="final_eval",
            paths=(Path("eval/final_eval/posterior_samples.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_matchups_manifest",
            description="Final-eval matchup manifest CSV",
            category="final_eval",
            paths=(Path("eval/final_eval/matchups.csv"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_payoff_matrix_export",
            description="Final-eval payoff matrix CSV export",
            category="final_eval",
            paths=(
                Path("eval/final_eval/payoff_matrices/p_mean.csv"),
                Path("eval/final_eval/matrices/mean.csv"),
            ),
        ),
        RequiredArtifactSpec(
            artifact_id="diagnostics_seat_bias",
            description="Seat-bias diagnostic JSON",
            category="diagnostics",
            paths=(Path("eval/diagnostics/seat_bias.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="diagnostics_truncation_heatmap",
            description="Truncation heatmap CSV",
            category="diagnostics",
            paths=(Path("eval/diagnostics/truncation_heatmap_data.csv"),),
        ),
        RequiredArtifactSpec(
            artifact_id="diagnostics_replay_verification",
            description="Replay verification JSON",
            category="diagnostics",
            paths=(Path("eval/diagnostics/replay_verification.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="sensitivity_summary",
            description="Sensitivity report summary JSON",
            category="sensitivity",
            paths=(Path("eval/final_eval/sensitivity/summary.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="paper_figures_pdf",
            description="Rendered paper figures (PDF)",
            category="figures",
            glob="figures/paper/*.pdf",
        ),
        RequiredArtifactSpec(
            artifact_id="paper_figures_png",
            description="Rendered paper figures (PNG)",
            category="figures",
            glob="figures/paper/*.png",
        ),
    ]
    for case_id in _REQUIRED_SENSITIVITY_CASE_IDS:
        specs.extend(
            (
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_summary",
                    description=f"Sensitivity {case_id} summary JSON",
                    category="sensitivity",
                    paths=(Path(f"eval/final_eval/sensitivity/{case_id}/summary.json"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_payoff_matchups",
                    description=f"Sensitivity {case_id} payoff matchup CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/final_eval/sensitivity/{case_id}/payoff/matchups.csv"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_nash_mixture",
                    description=f"Sensitivity {case_id} Nash mixture CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/final_eval/sensitivity/{case_id}/nash/mixture_mean.csv"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_alpharank_stationary",
                    description=f"Sensitivity {case_id} AlphaRank stationary CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/final_eval/sensitivity/{case_id}/alpharank/stationary_mean.csv"),),
                ),
            )
        )
    return tuple(specs)


def _evaluate_required_artifact(*, run_dir: Path, spec: RequiredArtifactSpec) -> dict[str, Any]:
    if spec.glob is not None:
        matches = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.glob(spec.glob) if path.is_file())
        passed = len(matches) >= spec.minimum_count
        return {
            "passed": passed,
            "category": spec.category,
            "description": spec.description,
            "glob": spec.glob,
            "minimum_count": spec.minimum_count,
            "matches": matches,
        }

    candidates = [path.as_posix() for path in spec.paths]
    for candidate in spec.paths:
        resolved = run_dir / candidate
        if resolved.is_file():
            return {
                "passed": True,
                "category": spec.category,
                "description": spec.description,
                "expected_paths": candidates,
                "resolved_path": candidate.as_posix(),
            }
    return {
        "passed": False,
        "category": spec.category,
        "description": spec.description,
        "expected_paths": candidates,
        "resolved_path": None,
    }


def _build_manifest_contract(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _load_json_object(manifest_path)
    except Exception as exc:
        return {
            "passed": False,
            "manifest_path": manifest_path.as_posix(),
            "fields": {},
            "consistency_checks": {},
            "missing_fields": [],
            "invalid_fields": [],
            "mismatches": [],
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }

    field_checks = {
        "run_id256": _validate_hex_field(manifest.get("run_id256"), length=64),
        "run_id64": _validate_hex_field(manifest.get("run_id64"), length=16),
        "git_commit": _validate_hex_field(manifest.get("git_commit"), length=40),
        "git_dirty": _validate_bool_field(manifest.get("git_dirty")),
        "spec_hash256": _validate_hex_field(manifest.get("spec_hash256"), length=64),
        "config_hash256": _validate_hex_field(manifest.get("config_hash256"), length=64),
        "simulator": _validate_simulator_manifest(manifest.get("simulator")),
        "spec_bundle": _validate_object_field(manifest.get("spec_bundle"), require_non_empty=True),
        "config_canonical": _validate_object_field(manifest.get("config_canonical"), require_non_empty=True),
        "seed_files": _validate_seed_files_field(manifest.get("seed_files")),
        "hardware": _validate_object_field(manifest.get("hardware"), require_non_empty=True),
        "evaluation_pinning": _validate_object_field(manifest.get("evaluation_pinning"), require_non_empty=True),
        "policy_set_selection": _validate_manifest_policy_set_selection(
            manifest.get("policy_set_selection"),
            details=manifest.get("policy_set_selection_details"),
        ),
    }
    missing_fields = [name for name, result in field_checks.items() if result["reason"] == "missing"]
    invalid_fields = [
        name
        for name, result in field_checks.items()
        if not result["passed"] and result["reason"] != "missing"
    ]

    consistency_checks = {
        "spec_bundle_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "spec_bundle.json",
            expected=manifest.get("spec_bundle"),
        ),
        "config_canonical_json_matches_manifest": _compare_json_file_to_manifest(
            file_path=run_dir / "config_canonical.json",
            expected=manifest.get("config_canonical"),
        ),
        "spec_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "spec_hash256.txt",
            expected=manifest.get("spec_hash256"),
        ),
        "config_hash_file_matches_manifest": _compare_text_file_to_manifest(
            file_path=run_dir / "config_hash256.txt",
            expected=manifest.get("config_hash256"),
        ),
    }
    mismatches = [name for name, result in consistency_checks.items() if not result["passed"]]
    passed = not missing_fields and not invalid_fields and not mismatches

    return {
        "passed": passed,
        "manifest_path": manifest_path.as_posix(),
        "fields": field_checks,
        "consistency_checks": consistency_checks,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "mismatches": mismatches,
        "message": (
            "manifest satisfies paper-readiness requirements"
            if passed
            else "manifest is missing required fields or has inconsistent companion files"
        ),
    }


def _build_final_eval_artifact_contract(final_eval_dir: Path) -> dict[str, Any]:
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
    if len(set(policy_ids)) != len(policy_ids):
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
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

    expected_keys = {(left, right) for left in range(len(policy_ids)) for right in range(left, len(policy_ids))}
    observed_keys: dict[tuple[int, int], str] = {}
    duplicate_matchups: list[str] = []
    noncanonical_matchups: list[str] = []
    reference_failures: list[str] = []
    policy_set_check = _validate_final_eval_policy_set(final_eval_dir=final_eval_dir, policy_ids=policy_ids)
    sensitivity_check = _validate_sensitivity_summary(final_eval_dir=final_eval_dir, policy_ids=policy_ids)

    try:
        for index, matchup in enumerate(_matchups(payload)):
            focal_index = _matchup_policy_index(
                matchup,
                index_field="focal_policy_index",
                policy_field="focal_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            opponent_index = _matchup_policy_index(
                matchup,
                index_field="opponent_policy_index",
                policy_field="opponent_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            pair_label = f"{policy_ids[focal_index]}__vs__{policy_ids[opponent_index]}"
            if focal_index > opponent_index:
                noncanonical_matchups.append(pair_label)
            key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
            if key in observed_keys:
                duplicate_matchups.append(pair_label)
            else:
                observed_keys[key] = pair_label

            for field_name, expected_kind in (
                ("matchup_dir", "directory"),
                ("episodes_path", "file"),
                ("summary_path", "file"),
                ("diagnostics_path", "file"),
                ("posterior_samples_path", "file"),
            ):
                try:
                    artifact_path = _require_relative_artifact_path(
                        final_eval_dir,
                        value=matchup.get(field_name),
                        field_name=f"matchups[{index}].{field_name}",
                    )
                except ValueError as exc:
                    reference_failures.append(str(exc))
                    continue
                exists = artifact_path.is_dir() if expected_kind == "directory" else artifact_path.is_file()
                if not exists:
                    reference_failures.append(
                        "matchups["
                        f"{index}].{field_name} missing {expected_kind}: "
                        f"{artifact_path.relative_to(final_eval_dir).as_posix()}"
                    )
    except ValueError as exc:
        return {
            "passed": False,
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "expected_matchup_count": len(expected_keys),
            "observed_matchup_count": len(observed_keys),
            "missing_matchups": [],
            "duplicate_matchups": duplicate_matchups,
            "noncanonical_matchups": noncanonical_matchups,
            "reference_failures": [str(exc)],
            "policy_set": policy_set_check,
            "sensitivity_summary": sensitivity_check,
            "reason": "invalid_matchup_index",
            "message": str(exc),
        }

    missing_matchups = [
        f"{policy_ids[left]}__vs__{policy_ids[right]}"
        for left, right in sorted(expected_keys - set(observed_keys))
    ]

    passed = not duplicate_matchups and not noncanonical_matchups and not missing_matchups and not reference_failures
    passed = passed and bool(policy_set_check["passed"]) and bool(sensitivity_check["passed"])

    return {
        "passed": passed,
        "summary_path": summary_path.as_posix(),
        "policy_ids": list(policy_ids),
        "expected_matchup_count": len(expected_keys),
        "observed_matchup_count": len(observed_keys),
        "missing_matchups": missing_matchups,
        "duplicate_matchups": duplicate_matchups,
        "noncanonical_matchups": noncanonical_matchups,
        "reference_failures": reference_failures,
        "policy_set": policy_set_check,
        "sensitivity_summary": sensitivity_check,
        "message": (
            "final_eval artifact contract is complete"
            if passed
            else "final_eval artifact contract is missing required referenced artifacts"
        ),
    }


def _validate_final_eval_policy_set(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
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


def _validate_sensitivity_summary(*, final_eval_dir: Path, policy_ids: Sequence[str]) -> dict[str, Any]:
    summary_path = final_eval_dir / "sensitivity" / "summary.json"
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
    missing_cases = [case_id for case_id in _REQUIRED_SENSITIVITY_CASE_IDS if case_id not in raw_cases]
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


def _build_final_eval_guardrail_summary(
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
    summary_path = final_eval_dir / "summary.json"
    payload = _load_json_object(summary_path)
    policy_ids = _policy_ids(payload)
    matchups = _matchups(payload)
    canonical_matchups = _canonical_unordered_matchups(matchups, policy_ids=policy_ids)
    matchup_diagnostics = _load_matchup_diagnostics(final_eval_dir=final_eval_dir, matchups=canonical_matchups)

    truncation = _build_truncation_check(
        matchup_diagnostics,
        max_truncation_rate=max_truncation_rate,
    )
    seat_bias = _build_seat_bias_check(
        matchup_diagnostics=matchup_diagnostics,
        max_abs_delta=seat_bias_max_abs_delta,
        posterior_min=seat_bias_posterior_min,
    )
    baseline = _build_baseline_check(
        payload,
        policy_ids=policy_ids,
        focal_policy_id=focal_policy_id,
        baseline_policy_id=baseline_policy_id,
        win_rate_threshold=baseline_win_rate_threshold,
        posterior_min=baseline_posterior_min,
    )

    checks = {
        "truncation_rate": truncation,
        "seat_bias_alarm": seat_bias,
        "baseline_win_rate_vs_b0": baseline,
    }
    alarms = [name for name, check in checks.items() if not bool(check["passed"])]
    metadata = cast(Mapping[str, Any], payload.get("metadata", {}))

    return {
        "passed": not alarms,
        "alarms": alarms,
        "final_eval": {
            "dir": final_eval_dir.as_posix(),
            "summary_path": summary_path.as_posix(),
            "policy_ids": list(policy_ids),
            "selection": dict(cast(Mapping[str, Any], metadata.get("selection", {}))),
        },
        "checks": checks,
    }


def _build_truncation_check(
    matchup_diagnostics: Sequence[Mapping[str, Any]],
    *,
    max_truncation_rate: float,
) -> dict[str, Any]:
    total_games = sum(_as_int(matchup["total_games"], context="total_games") for matchup in matchup_diagnostics)
    truncated_games = sum(_as_int(matchup["truncations"], context="truncations") for matchup in matchup_diagnostics)
    rate = (truncated_games / total_games) if total_games else None
    passed = total_games > 0 and rate is not None and rate <= max_truncation_rate
    result: dict[str, Any] = {
        "passed": passed,
        "truncated_games": truncated_games,
        "total_games": total_games,
        "rate": rate,
        "max_allowed_rate": max_truncation_rate,
    }
    if total_games == 0:
        result["reason"] = "final_eval_summary_contains_no_games"
    return result


def _build_seat_bias_check(
    *,
    matchup_diagnostics: Sequence[Mapping[str, Any]],
    max_abs_delta: float,
    posterior_min: float,
) -> dict[str, Any]:
    per_matchup: list[dict[str, Any]] = []
    seat0_wins = 0
    seat1_wins = 0
    draws = 0
    truncations = 0
    engine_errors = 0

    for matchup in matchup_diagnostics:
        matchup_seat0_wins = _as_int(matchup["seat0_wins"], context="seat0_wins")
        matchup_seat1_wins = _as_int(matchup["seat1_wins"], context="seat1_wins")
        matchup_draws = _as_int(matchup["draws"], context="draws")
        matchup_truncations = _as_int(matchup["truncations"], context="truncations")
        matchup_engine_errors = _as_int(matchup["engine_errors"], context="engine_errors")
        decisive_games = _as_int(matchup["decisive_games"], context="decisive_games")

        seat0_wins += matchup_seat0_wins
        seat1_wins += matchup_seat1_wins
        draws += matchup_draws
        truncations += matchup_truncations
        engine_errors += matchup_engine_errors

        per_matchup.append(
            {
                "focal_policy_id": str(matchup["focal_policy_id"]),
                "opponent_policy_id": str(matchup["opponent_policy_id"]),
                "diagnostics_path": str(matchup["diagnostics_path"]),
                "seat0_wins": matchup_seat0_wins,
                "seat1_wins": matchup_seat1_wins,
                "decisive_games": decisive_games,
                "seat0_win_rate": (matchup_seat0_wins / decisive_games) if decisive_games else None,
                "seat1_win_rate": (matchup_seat1_wins / decisive_games) if decisive_games else None,
                "draws": matchup_draws,
                "truncations": matchup_truncations,
                "engine_errors": matchup_engine_errors,
            }
        )

    decisive_games = seat0_wins + seat1_wins
    result: dict[str, Any] = {
        "passed": False,
        "alarm": None,
        "observed": {
            "seat0_wins": seat0_wins,
            "seat1_wins": seat1_wins,
            "draws": draws,
            "truncations": truncations,
            "engine_errors": engine_errors,
            "decisive_games": decisive_games,
            "total_games": decisive_games + draws + truncations,
        },
        "thresholds": {
            "max_abs_delta_from_half": max_abs_delta,
            "posterior_probability": posterior_min,
        },
        "per_matchup": per_matchup,
    }
    if decisive_games == 0:
        result["reason"] = "seat_bias_requires_at_least_one_decisive_game"
        return result

    alpha = seat0_wins + 0.5
    beta_param = seat1_wins + 0.5
    ci_low, ci_high = beta_dist.ppf((0.025, 0.975), alpha, beta_param)
    prob_gt_upper = 1.0 - float(beta_dist.cdf(0.5 + max_abs_delta, alpha, beta_param))
    prob_lt_lower = float(beta_dist.cdf(0.5 - max_abs_delta, alpha, beta_param))
    alarm = prob_gt_upper > posterior_min or prob_lt_lower > posterior_min

    result["passed"] = not alarm
    result["alarm"] = alarm
    result["posterior"] = {
        "mean": float(alpha / (alpha + beta_param)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_gt_half_plus_delta": prob_gt_upper,
        "prob_lt_half_minus_delta": prob_lt_lower,
    }
    return result


def _build_baseline_check(
    payload: Mapping[str, Any],
    *,
    policy_ids: Sequence[str],
    focal_policy_id: str | None,
    baseline_policy_id: str,
    win_rate_threshold: float,
    posterior_min: float,
) -> dict[str, Any]:
    resolved_focal_policy_id = focal_policy_id
    focal_policy_source = "explicit_arg" if focal_policy_id is not None else None
    inferred_eligible_policy_ids: list[str] | None = None

    if resolved_focal_policy_id is None:
        inferred = _infer_focal_policy_id(
            payload,
            policy_ids,
            baseline_policy_id=baseline_policy_id,
        )
        resolved_focal_policy_id = cast(str | None, inferred["focal_policy_id"])
        focal_policy_source = cast(str | None, inferred["source"])
        inferred_eligible_policy_ids = cast(list[str] | None, inferred.get("eligible_non_baseline_policy_ids"))

    result: dict[str, Any] = {
        "passed": False,
        "baseline_policy_id": baseline_policy_id,
        "focal_policy_id": resolved_focal_policy_id,
        "focal_policy_source": focal_policy_source,
        "win_rate_threshold": win_rate_threshold,
        "posterior_probability_threshold": posterior_min,
    }
    if inferred_eligible_policy_ids is not None:
        result["eligible_non_baseline_policy_ids"] = inferred_eligible_policy_ids

    if baseline_policy_id not in policy_ids:
        result["reason"] = "baseline_policy_missing_from_final_eval"
        return result
    if resolved_focal_policy_id is None:
        if inferred_eligible_policy_ids:
            result["reason"] = "ambiguous_non_baseline_focal_policy"
            result["message"] = (
                "multiple eligible non-baseline policies found; "
                "pass --focal-policy-id to choose the focal policy explicitly"
            )
        else:
            result["reason"] = "could_not_infer_non_baseline_focal_policy"
        return result
    if resolved_focal_policy_id not in policy_ids:
        result["reason"] = "focal_policy_missing_from_final_eval"
        return result
    if resolved_focal_policy_id == baseline_policy_id:
        result["reason"] = "focal_policy_matches_baseline_policy"
        return result

    focal_index = policy_ids.index(resolved_focal_policy_id)
    baseline_index = policy_ids.index(baseline_policy_id)
    posterior_samples = _posterior_samples(payload, focal_index=focal_index, opponent_index=baseline_index)
    has_payoff_samples = bool(_matrix_cell(payload, field="has_payoff_samples", row=focal_index, column=baseline_index))
    mean = _as_optional_float(_matrix_cell(payload, field="mean", row=focal_index, column=baseline_index))
    ci_low = _as_optional_float(_matrix_cell(payload, field="ci_low", row=focal_index, column=baseline_index))
    ci_high = _as_optional_float(_matrix_cell(payload, field="ci_high", row=focal_index, column=baseline_index))
    paired_seed_count = _as_int(
        _matrix_cell(payload, field="paired_seed_count", row=focal_index, column=baseline_index),
        context="paired_seed_count",
    )
    stop_reason = str(_matrix_cell(payload, field="stop_reason", row=focal_index, column=baseline_index))
    prob_gt_threshold = (
        sum(1 for sample in posterior_samples if sample > win_rate_threshold) / len(posterior_samples)
        if posterior_samples
        else None
    )

    result.update(
        {
            "has_payoff_samples": has_payoff_samples,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "paired_seed_count": paired_seed_count,
            "stop_reason": stop_reason,
            "sample_count": len(posterior_samples),
            "prob_gt_threshold": prob_gt_threshold,
        }
    )

    if not has_payoff_samples or mean is None or prob_gt_threshold is None:
        result["reason"] = "baseline_matchup_has_no_payoff_samples"
        return result

    result["passed"] = prob_gt_threshold >= posterior_min
    return result


def _infer_focal_policy_id(
    payload: Mapping[str, Any],
    policy_ids: Sequence[str],
    *,
    baseline_policy_id: str,
) -> dict[str, Any]:
    metadata_focal_policy_id = _metadata_focal_policy_id(payload)
    if metadata_focal_policy_id is not None:
        return {
            "focal_policy_id": metadata_focal_policy_id,
            "source": "metadata",
        }

    baseline_ids = {
        RANDOM_LEGAL_POLICY_ID,
        NO_LEAGUE_POLICY_ID,
        HEURISTIC_PUBLIC_POLICY_ID,
        baseline_policy_id,
    }
    eligible_policy_ids = [policy_id for policy_id in policy_ids if policy_id not in baseline_ids]
    if len(eligible_policy_ids) == 1:
        return {
            "focal_policy_id": eligible_policy_ids[0],
            "source": "sole_eligible_non_baseline",
        }
    return {
        "focal_policy_id": None,
        "source": None,
        "eligible_non_baseline_policy_ids": eligible_policy_ids,
    }


def _policy_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_policy_ids = payload.get("policy_ids")
    if not isinstance(raw_policy_ids, list) or any(not isinstance(item, str) for item in raw_policy_ids):
        raise ValueError("final_eval summary must include string policy_ids")
    return list(raw_policy_ids)


def _matchups(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include matchups")
    matchups: list[Mapping[str, Any]] = []
    for index, matchup in enumerate(raw_matchups):
        matchups.append(_mapping(matchup, context=f"matchups[{index}]"))
    return matchups


def _canonical_unordered_matchups(
    matchups: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected: dict[tuple[int, int], tuple[int, Mapping[str, Any]]] = {}
    for index, matchup in enumerate(matchups):
        focal_index = _matchup_policy_index(
            matchup,
            index_field="focal_policy_index",
            policy_field="focal_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        opponent_index = _matchup_policy_index(
            matchup,
            index_field="opponent_policy_index",
            policy_field="opponent_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
        rank = 0 if focal_index <= opponent_index else 1
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, matchup)
    return [selected[key][1] for key in sorted(selected)]


def _load_matchup_diagnostics(
    *,
    final_eval_dir: Path,
    matchups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for matchup in matchups:
        diagnostics_path = final_eval_dir / str(matchup["diagnostics_path"])
        diagnostics = _load_json_object(diagnostics_path)
        seat_results = _mapping(diagnostics.get("seat_results"), context=f"{diagnostics_path}:seat_results")
        seat0_wins = _as_int(seat_results.get("seat0_wins"), context=f"{diagnostics_path}:seat0_wins")
        seat1_wins = _as_int(seat_results.get("seat1_wins"), context=f"{diagnostics_path}:seat1_wins")
        draws = _as_int(seat_results.get("draws"), context=f"{diagnostics_path}:draws")
        truncations = _as_int(seat_results.get("truncations"), context=f"{diagnostics_path}:truncations")
        engine_errors = _as_int(seat_results.get("engine_errors"), context=f"{diagnostics_path}:engine_errors")
        decisive_games = seat0_wins + seat1_wins
        payloads.append(
            {
                "focal_policy_id": str(matchup["focal_policy_id"]),
                "opponent_policy_id": str(matchup["opponent_policy_id"]),
                "diagnostics_path": str(matchup["diagnostics_path"]),
                "seat0_wins": seat0_wins,
                "seat1_wins": seat1_wins,
                "draws": draws,
                "truncations": truncations,
                "engine_errors": engine_errors,
                "decisive_games": decisive_games,
                "total_games": decisive_games + draws + truncations,
            }
        )
    return payloads


def _matchup_policy_index(
    matchup: Mapping[str, Any],
    *,
    index_field: str,
    policy_field: str,
    policy_ids: Sequence[str],
    context: str,
) -> int:
    raw_index = matchup.get(index_field)
    if raw_index is not None:
        index = _as_int(raw_index, context=f"{context}.{index_field}")
        if index < 0 or index >= len(policy_ids):
            raise ValueError(
                f"{context}.{index_field}={index} is out of range for policy_ids with length {len(policy_ids)}"
            )
        return index
    policy_id = matchup.get(policy_field)
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"{context}.{policy_field} must be a non-empty string")
    try:
        return policy_ids.index(policy_id)
    except ValueError as exc:
        raise ValueError(f"{context}.{policy_field}={policy_id!r} is missing from policy_ids") from exc


def _metadata_focal_policy_id(payload: Mapping[str, Any]) -> str | None:
    metadata = _mapping(payload.get("metadata", {}), context="metadata")
    for path in (
        ("focal_policy_id",),
        ("recommended_focal_policy_id",),
        ("focal_policy", "policy_id"),
        ("selection", "focal_policy_id"),
    ):
        value = _nested_optional_string(metadata, path=path)
        if value is not None:
            return value
    return None


def _nested_optional_string(payload: Mapping[str, Any], *, path: Sequence[str]) -> str | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, str):
        normalized = current.strip()
        return normalized or None
    return None


def _matrix_cell(payload: Mapping[str, Any], *, field: str, row: int, column: int) -> Any:
    matrix = _matrix(payload, field=field)
    try:
        matrix_row = matrix[row]
        if not isinstance(matrix_row, list):
            raise TypeError
        return matrix_row[column]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"matrix {field!r} is missing cell [{row}][{column}]") from exc


def _matrix(payload: Mapping[str, Any], *, field: str) -> list[Any]:
    matrices = _mapping(payload.get("matrices"), context="matrices")
    matrix_payload = _mapping(matrices.get(field), context=f"matrices.{field}")
    values = matrix_payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"matrices.{field}.values must be a list")
    return values


def _posterior_samples(payload: Mapping[str, Any], *, focal_index: int, opponent_index: int) -> list[float]:
    posterior_payload = _mapping(payload.get("posterior_samples"), context="posterior_samples")
    values = posterior_payload.get("values")
    if not isinstance(values, list):
        raise ValueError("posterior_samples.values must be a list")
    try:
        row = values[focal_index]
        if not isinstance(row, list):
            raise TypeError
        samples = row[opponent_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"posterior_samples.values is missing cell [{focal_index}][{opponent_index}]"
        ) from exc
    if not isinstance(samples, list):
        raise ValueError("posterior sample cell must be a list")
    return [float(sample) for sample in samples]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], payload)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _as_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric matrix cell or null")
    return float(value)


def _validate_hex_field(value: Any, *, length: int) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, str):
        return {"passed": False, "reason": "invalid_type", "message": "field must be a string"}
    normalized = value.strip().lower()
    if len(normalized) != length or any(char not in "0123456789abcdef" for char in normalized):
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": f"field must be a {length}-character hex string",
        }
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_bool_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, bool):
        return {"passed": False, "reason": "invalid_type", "message": "field must be a boolean"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_object_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict):
        return {"passed": False, "reason": "invalid_type", "message": "field must be an object"}
    if require_non_empty and not value:
        return {"passed": False, "reason": "empty", "message": "field must not be empty"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_string_list_field(value: Any, *, require_non_empty: bool) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "field must be a list of non-empty strings",
        }
    if require_non_empty and not value:
        return {"passed": False, "reason": "empty", "message": "field must not be empty"}
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_manifest_policy_set_selection(value: Any, *, details: Any) -> dict[str, Any]:
    selection_check = _validate_string_list_field(value, require_non_empty=False)
    if not selection_check["passed"]:
        return selection_check
    if value:
        return {"passed": True, "reason": None, "message": "ok"}
    if _documents_unresolved_policy_set_selection(details):
        return {
            "passed": True,
            "reason": None,
            "message": "ok (policy_set_selection is unresolved but documented)",
        }
    return {
        "passed": False,
        "reason": "empty",
        "message": (
            "field must not be empty unless policy_set_selection_details documents an unresolved selection"
        ),
    }


def _documents_unresolved_policy_set_selection(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    if details.get("status") != "unresolved":
        return False
    reason = details.get("reason")
    if isinstance(reason, str) and reason.strip():
        return True
    missing_inputs = details.get("missing_inputs")
    return isinstance(missing_inputs, list) and any(
        isinstance(item, str) and item.strip() for item in missing_inputs
    )


def _validate_seed_files_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict) or not value:
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "seed_files must be a non-empty object",
        }
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            return {
                "passed": False,
                "reason": "invalid_key",
                "message": "seed_files keys must be non-empty strings",
            }
        if not isinstance(item, dict):
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": "seed_files entries must be objects",
            }
        path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not path.strip():
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": f"seed_files[{key!r}] must include a non-empty path",
            }
        hash_check = _validate_hex_field(sha256, length=64)
        if not hash_check["passed"]:
            return {
                "passed": False,
                "reason": "invalid_value",
                "message": f"seed_files[{key!r}] must include a 64-character hex sha256",
            }
    return {"passed": True, "reason": None, "message": "ok"}


def _validate_simulator_manifest(value: Any) -> dict[str, Any]:
    if value is None:
        return {"passed": False, "reason": "missing", "message": "field is missing"}
    if not isinstance(value, dict) or not value:
        return {
            "passed": False,
            "reason": "invalid_type",
            "message": "simulator must be a non-empty object",
        }
    version = value.get("version")
    if not isinstance(version, str) or not version.strip():
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": "simulator must include a non-empty version",
        }
    build_keys = ("compatibility_hash", "build", "build_id", "build_info", "commit", "sha256")
    if not any(isinstance(value.get(key), str) and str(value.get(key)).strip() for key in build_keys):
        return {
            "passed": False,
            "reason": "invalid_value",
            "message": "simulator must include build/version identity information",
        }
    return {"passed": True, "reason": None, "message": "ok"}


def _compare_json_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
    try:
        payload = _load_json_object(file_path)
    except Exception as exc:
        return {
            "passed": False,
            "file_path": file_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    passed = payload == expected
    return {
        "passed": passed,
        "file_path": file_path.as_posix(),
        "message": "JSON file matches manifest" if passed else "JSON file does not match manifest",
    }


def _compare_text_file_to_manifest(*, file_path: Path, expected: Any) -> dict[str, Any]:
    try:
        observed = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        return {
            "passed": False,
            "file_path": file_path.as_posix(),
            "reason": exc.__class__.__name__,
            "message": str(exc),
        }
    passed = isinstance(expected, str) and observed == expected
    return {
        "passed": passed,
        "file_path": file_path.as_posix(),
        "observed": observed,
        "message": "text file matches manifest" if passed else "text file does not match manifest",
    }


def _require_relative_artifact_path(root: Path, *, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path string")
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError(f"{field_name} must be relative to {root.as_posix()}")
    resolved_root = root.resolve()
    resolved_path = (root / raw_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} resolves outside {root.as_posix()}: {value}") from exc
    return resolved_path
