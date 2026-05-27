"""Paper-readiness auditing over run directories and final-eval artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.paper_readiness_fields import (
    compare_json_file_to_manifest as _compare_json_file_to_manifest,
)
from weiss_rl.eval.paper_readiness_fields import (
    compare_text_file_to_manifest as _compare_text_file_to_manifest,
)
from weiss_rl.eval.paper_readiness_fields import (
    load_json_object as _load_json_object,
)
from weiss_rl.eval.paper_readiness_fields import (
    require_relative_artifact_path as _require_relative_artifact_path,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_bool_field as _validate_bool_field,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_existing_file as _validate_existing_file,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_hex_field as _validate_hex_field,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_manifest_policy_set_selection as _validate_manifest_policy_set_selection,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_object_field as _validate_object_field,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_seed_files_field as _validate_seed_files_field,
)
from weiss_rl.eval.paper_readiness_fields import (
    validate_simulator_manifest as _validate_simulator_manifest,
)
from weiss_rl.eval.paper_readiness_guardrails import (
    build_final_eval_guardrail_summary as _build_final_eval_guardrail_summary,
)
from weiss_rl.eval.paper_readiness_guardrails import (
    matchup_policy_index as _matchup_policy_index,
)
from weiss_rl.eval.paper_readiness_guardrails import (
    matchups as _matchups,
)
from weiss_rl.eval.paper_readiness_guardrails import (
    policy_ids as _policy_ids,
)
from weiss_rl.eval.policy_set import RANDOM_LEGAL_POLICY_ID

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
            artifact_id="environment_json",
            description="Environment manifest JSON",
            category="run_root",
            paths=(Path("environment.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="run_summary_json",
            description="Run summary JSON",
            category="run_root",
            paths=(Path("run_summary.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="determinism_report_json",
            description="Determinism report JSON",
            category="run_root",
            paths=(Path("determinism_report.json"),),
        ),
        RequiredArtifactSpec(
            artifact_id="training_metrics",
            description="Training metrics JSONL or checkpoint interpolation provenance",
            category="training",
            paths=(
                Path("training/logs/training_metrics.jsonl"),
                Path("eval/diagnostics/checkpoint_interpolation_summary.json"),
            ),
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
            paths=(
                Path("eval/final_eval/posterior_samples.json"),
                Path("eval/final_eval/posterior_samples.npz"),
            ),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_matchups_manifest",
            description="Final-eval matchup manifest CSV",
            category="final_eval",
            paths=(Path("eval/final_eval/matchups.csv"),),
        ),
        RequiredArtifactSpec(
            artifact_id="final_eval_artifact_hashes",
            description="Final-eval artifact hashes JSON",
            category="final_eval",
            paths=(Path("eval/final_eval/artifact_hashes.json"),),
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
            paths=(
                Path("eval/metagame/summary.json"),
                Path("eval/final_eval/sensitivity/summary.json"),
            ),
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
                    paths=(
                        Path(f"eval/metagame/{case_id}/summary.json"),
                        Path(f"eval/final_eval/sensitivity/{case_id}/summary.json"),
                    ),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_payoff_matchups",
                    description=f"Sensitivity {case_id} payoff matchup CSV",
                    category="sensitivity",
                    paths=(
                        Path(f"eval/metagame/{case_id}/payoff/matchups.csv"),
                        Path(f"eval/final_eval/sensitivity/{case_id}/payoff/matchups.csv"),
                    ),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_nash_mixture",
                    description=f"Sensitivity {case_id} Nash mixture CSV",
                    category="sensitivity",
                    paths=(
                        Path(f"eval/metagame/{case_id}/nash/mixture_mean.csv"),
                        Path(f"eval/final_eval/sensitivity/{case_id}/nash/mixture_mean.csv"),
                    ),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_alpharank_stationary",
                    description=f"Sensitivity {case_id} AlphaRank stationary CSV",
                    category="sensitivity",
                    paths=(
                        Path(f"eval/metagame/{case_id}/alpharank/stationary_mean.csv"),
                        Path(f"eval/final_eval/sensitivity/{case_id}/alpharank/stationary_mean.csv"),
                    ),
                ),
            )
        )
    return tuple(specs)


def _sensitivity_root_candidates(*, final_eval_dir: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    canonical = final_eval_dir.parent / "metagame"
    legacy = final_eval_dir / "sensitivity"
    for candidate in (canonical, legacy):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _resolve_sensitivity_summary_path(final_eval_dir: Path) -> Path:
    for candidate_root in _sensitivity_root_candidates(final_eval_dir=final_eval_dir):
        summary_path = candidate_root / "summary.json"
        if summary_path.is_file():
            return summary_path
    return _sensitivity_root_candidates(final_eval_dir=final_eval_dir)[0] / "summary.json"


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
    layout = ArtifactLayout.from_run_dir(run_dir)
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
        name for name, result in field_checks.items() if not result["passed"] and result["reason"] != "missing"
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
        "run_summary_exists": _validate_existing_file(layout.run_summary_path),
        "environment_manifest_exists": _validate_existing_file(layout.environment_path),
        "determinism_report_exists": _validate_existing_file(layout.determinism_report_path),
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
        f"{policy_ids[left]}__vs__{policy_ids[right]}" for left, right in sorted(expected_keys - set(observed_keys))
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
    summary_path = _resolve_sensitivity_summary_path(final_eval_dir)
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
