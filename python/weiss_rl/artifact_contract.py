"""Declarative artifact contract for thesis run trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANONICAL_RUN_ROOT_FILES: tuple[Path, ...] = (
    Path("manifest.json"),
    Path("environment.json"),
    Path("run_summary.json"),
    Path("paper_readiness_summary.json"),
    Path("determinism_report.json"),
)

CANONICAL_RUN_TREE_DIRS: tuple[Path, ...] = (
    Path("training"),
    Path("eval/final_eval"),
    Path("eval/diagnostics"),
    Path("eval/metagame"),
    Path("replays"),
    Path("figures/paper"),
)

REQUIRED_SENSITIVITY_CASE_IDS = ("S0", "S1", "S2")


@dataclass(frozen=True, slots=True)
class RequiredArtifactSpec:
    artifact_id: str
    description: str
    category: str
    paths: tuple[Path, ...] = ()
    compatibility_paths: tuple[Path, ...] = ()
    glob: str | None = None
    minimum_count: int = 1

    def candidate_paths(self) -> tuple[Path, ...]:
        return (*self.paths, *self.compatibility_paths)


def required_run_artifact_specs() -> tuple[RequiredArtifactSpec, ...]:
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
            paths=(Path("eval/final_eval/payoff_matrices/p_mean.csv"),),
            compatibility_paths=(Path("eval/final_eval/matrices/mean.csv"),),
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
            paths=(Path("eval/metagame/summary.json"),),
            compatibility_paths=(Path("eval/final_eval/sensitivity/summary.json"),),
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
    for case_id in REQUIRED_SENSITIVITY_CASE_IDS:
        specs.extend(
            (
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_summary",
                    description=f"Sensitivity {case_id} summary JSON",
                    category="sensitivity",
                    paths=(Path(f"eval/metagame/{case_id}/summary.json"),),
                    compatibility_paths=(Path(f"eval/final_eval/sensitivity/{case_id}/summary.json"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_payoff_matchups",
                    description=f"Sensitivity {case_id} payoff matchup CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/metagame/{case_id}/payoff/matchups.csv"),),
                    compatibility_paths=(Path(f"eval/final_eval/sensitivity/{case_id}/payoff/matchups.csv"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_nash_mixture",
                    description=f"Sensitivity {case_id} Nash mixture CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/metagame/{case_id}/nash/mixture_mean.csv"),),
                    compatibility_paths=(Path(f"eval/final_eval/sensitivity/{case_id}/nash/mixture_mean.csv"),),
                ),
                RequiredArtifactSpec(
                    artifact_id=f"sensitivity_{case_id.lower()}_alpharank_stationary",
                    description=f"Sensitivity {case_id} AlphaRank stationary CSV",
                    category="sensitivity",
                    paths=(Path(f"eval/metagame/{case_id}/alpharank/stationary_mean.csv"),),
                    compatibility_paths=(Path(f"eval/final_eval/sensitivity/{case_id}/alpharank/stationary_mean.csv"),),
                ),
            )
        )
    return tuple(specs)


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


def evaluate_required_artifact(*, run_dir: Path, spec: RequiredArtifactSpec) -> dict[str, object]:
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

    candidate_paths = spec.candidate_paths()
    candidates = [path.as_posix() for path in candidate_paths]
    for candidate in candidate_paths:
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


def build_run_directory_audit(run_dir: Path) -> dict[str, object]:
    artifact_results = {
        spec.artifact_id: evaluate_required_artifact(run_dir=run_dir, spec=spec)
        for spec in required_run_artifact_specs()
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
