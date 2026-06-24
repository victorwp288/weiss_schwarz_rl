"""Required artifact inventory for paper-readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REQUIRED_SENSITIVITY_CASE_IDS = ("S0", "S1", "S2")


@dataclass(frozen=True, slots=True)
class RequiredArtifactGroup:
    category: str
    title: str
    purpose: str


@dataclass(frozen=True, slots=True)
class RequiredArtifactSpec:
    artifact_id: str
    description: str
    category: str
    paths: tuple[Path, ...] = ()
    glob: str | None = None
    minimum_count: int = 1


REQUIRED_ARTIFACT_GROUPS = (
    RequiredArtifactGroup(
        category="run_root",
        title="Run identity",
        purpose="Pins the run, config, environment, and determinism metadata used to interpret every output.",
    ),
    RequiredArtifactGroup(
        category="training",
        title="Training evidence",
        purpose="Shows how the selected policy was produced or, for an interpolation, which checkpoints were blended.",
    ),
    RequiredArtifactGroup(
        category="final_eval",
        title="Final evaluation",
        purpose="Stores the retained policy panel, matchup schedule, payoff matrix, posterior samples, and hashes.",
    ),
    RequiredArtifactGroup(
        category="diagnostics",
        title="Diagnostics",
        purpose="Captures seat bias, truncation, replay, and other checks that can invalidate a headline result.",
    ),
    RequiredArtifactGroup(
        category="sensitivity",
        title="Metagame sensitivity",
        purpose="Records S0-S2 robustness cases for payoff, Nash, and AlphaRank interpretations.",
    ),
    RequiredArtifactGroup(
        category="figures",
        title="Paper figures",
        purpose="Keeps the rendered figure outputs tied to the same run tree as the evaluation evidence.",
    ),
)


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
    for case_id in REQUIRED_SENSITIVITY_CASE_IDS:
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


def required_run_artifact_group_payload(
    specs: tuple[RequiredArtifactSpec, ...] | None = None,
) -> dict[str, dict[str, int | str]]:
    resolved_specs = specs if specs is not None else required_run_artifact_specs()
    counts = {group.category: 0 for group in REQUIRED_ARTIFACT_GROUPS}
    for spec in resolved_specs:
        counts[spec.category] = counts.get(spec.category, 0) + 1
    return {
        group.category: {
            "title": group.title,
            "purpose": group.purpose,
            "artifact_count": counts[group.category],
        }
        for group in REQUIRED_ARTIFACT_GROUPS
    }


__all__ = [
    "REQUIRED_ARTIFACT_GROUPS",
    "REQUIRED_SENSITIVITY_CASE_IDS",
    "RequiredArtifactGroup",
    "RequiredArtifactSpec",
    "required_run_artifact_group_payload",
    "required_run_artifact_specs",
]
