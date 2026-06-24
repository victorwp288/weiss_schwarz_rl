"""Named sections of the final-eval summary payload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import hash_seed_file
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.analysis.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.final.matrices import (
    build_final_eval_matrices,
    build_final_eval_posterior_samples,
    canonical_matchup_results_by_key,
    covered_matrix_cells,
)


@dataclass(frozen=True, slots=True)
class FinalEvalSummarySection:
    key: str
    question: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "question": self.question,
            "evidence": list(self.evidence),
        }


FINAL_EVAL_SUMMARY_SECTIONS: tuple[FinalEvalSummarySection, ...] = (
    FinalEvalSummarySection(
        key="metadata.selection",
        question="Which policies were evaluated, and why were they selected?",
        evidence=("policy_ids", "metadata.selection", "policy_set.json"),
    ),
    FinalEvalSummarySection(
        key="metadata.matchup_artifacts",
        question="How are unordered pair artifacts mapped into directed matrix cells?",
        evidence=("metadata.matchup_artifacts", "matchups", "matchups.csv"),
    ),
    FinalEvalSummarySection(
        key="matrices",
        question="What is the directed payoff estimate for every focal/opponent pair?",
        evidence=("matrices/mean", "matrices/ci_low", "matrices/ci_high", "matrices/wins"),
    ),
    FinalEvalSummarySection(
        key="posterior_samples",
        question="What posterior samples support the matrix estimates?",
        evidence=("posterior_samples.json", "posterior_samples.npz"),
    ),
    FinalEvalSummarySection(
        key="matchups",
        question="Which canonical matchup artifacts support each matrix cell?",
        evidence=("matchup_dir", "episodes_path", "diagnostics_path", "posterior_samples_path"),
    ),
)


def final_eval_summary_sections_payload() -> list[dict[str, object]]:
    return [section.as_payload() for section in FINAL_EVAL_SUMMARY_SECTIONS]


def build_final_eval_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    paired_seeds: Sequence[int],
    stop_rules: StopRulesConfig,
    scheme: PayoffFoldScheme,
    sample_count: int,
    selection_payload: Mapping[str, Any],
    seed_file_path: Path | None,
) -> dict[str, Any]:
    resolved = dict(metadata or {})
    resolved.update(
        {
            "policy_count": len(policy_ids),
            "matchup_count": len(matchup_results),
            "matchup_artifacts": {
                "kind": "canonical_unordered_pairs_v1",
                "canonical_order": "focal_policy_index <= opponent_policy_index",
                "reverse_matrix_cells": "derived_from_canonical_matchup_artifacts",
            },
            "stage1_paired_seeds": int(stage1_paired_seeds),
            "max_paired_seeds": int(max_paired_seeds),
            "paired_seed_budget": len(paired_seeds),
            "stop_rules": {
                "stop_delta_ci_half_width": float(stop_rules.stop_delta_ci_half_width),
                "stop_confidence": float(stop_rules.stop_confidence),
            },
            "scheme": scheme,
            "sample_count": int(sample_count),
            "selection": dict(selection_payload),
        }
    )
    if seed_file_path is not None:
        resolved["seed_file"] = {
            "path": seed_file_path.as_posix(),
            "sha256": hash_seed_file(seed_file_path),
        }
    return resolved


def build_final_eval_matchup_rows(
    *,
    output_dir: Path,
    matchup_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "focal_policy_id": result["focal_policy_id"],
            "opponent_policy_id": result["opponent_policy_id"],
            "focal_policy_index": result["focal_index"],
            "opponent_policy_index": result["opponent_index"],
            "matchup_dir": relative_to(result["matchup_dir"], root=output_dir),
            "episodes_path": relative_to(result["episodes_path"], root=output_dir),
            "summary_path": relative_to(Path(result["matchup_dir"]) / "matchup_summary.json", root=output_dir),
            "diagnostics_path": relative_to(Path(result["matchup_dir"]) / "diagnostics.json", root=output_dir),
            "posterior_samples_path": relative_to(
                Path(result["matchup_dir"]) / "posterior_samples.json",
                root=output_dir,
            ),
            "matrix_cells": covered_matrix_cells(
                focal_index=int(result["focal_index"]),
                opponent_index=int(result["opponent_index"]),
            ),
            "paired_seed_count": result["summary"]["paired_seeds"],
            "observed_paired_seed_count": result["summary"]["observed_paired_seeds"],
            "excluded_paired_seed_count": result["summary"]["excluded_paired_seeds"],
            "has_payoff_samples": result["summary"]["has_payoff_samples"],
            "stop_reason": result["summary"]["stop_reason"],
        }
        for result in matchup_results
    ]


def relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "FINAL_EVAL_SUMMARY_SECTIONS",
    "FinalEvalSummarySection",
    "build_final_eval_matchup_rows",
    "build_final_eval_matrices",
    "build_final_eval_metadata",
    "build_final_eval_posterior_samples",
    "canonical_matchup_results_by_key",
    "final_eval_summary_sections_payload",
    "relative_to",
]
