"""Final-eval top-level payload assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import hash_seed_file
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.final.matrices import MATRIX_FIELDS, build_matrix, covered_matrix_cells, posterior_samples_cell
from weiss_rl.eval.payoff_folding import PayoffFoldScheme


def build_final_eval_payload(
    *,
    output_dir: Path,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    paired_seeds: Sequence[int],
    stop_rules: StopRulesConfig,
    scheme: PayoffFoldScheme,
    sample_count: int,
    selection_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    seed_file_path: Path | None,
) -> dict[str, Any]:
    canonical_results_by_key = {
        (int(result["focal_index"]), int(result["opponent_index"])): result for result in matchup_results
    }
    matrices = {
        field: build_matrix(
            policy_ids=policy_ids,
            canonical_results_by_key=canonical_results_by_key,
            field=field,
        )
        for field in MATRIX_FIELDS
    }
    posterior_matrix = [
        [
            posterior_samples_cell(
                canonical_results_by_key=canonical_results_by_key,
                focal_index=focal_index,
                opponent_index=opponent_index,
            )
            for opponent_index, _opponent_policy_id in enumerate(policy_ids)
        ]
        for focal_index, _focal_policy_id in enumerate(policy_ids)
    ]
    top_level_metadata = dict(metadata or {})
    top_level_metadata.update(
        {
            "policy_count": len(policy_ids),
            "matchup_count": len(matchup_results),
            "matchup_artifacts": {
                "kind": "canonical_unordered_pairs_v1",
                "canonical_order": "focal_policy_index <= opponent_policy_index",
                "reverse_matrix_cells": "derived_from_canonical_matchup_artifacts",
            },
            "stage1_paired_seeds": stage1_paired_seeds,
            "max_paired_seeds": max_paired_seeds,
            "paired_seed_budget": len(paired_seeds),
            "stop_rules": {
                "stop_delta_ci_half_width": float(stop_rules.stop_delta_ci_half_width),
                "stop_confidence": float(stop_rules.stop_confidence),
            },
            "scheme": scheme,
            "sample_count": sample_count,
            "selection": dict(selection_payload),
        }
    )
    if seed_file_path is not None:
        top_level_metadata["seed_file"] = {
            "path": seed_file_path.as_posix(),
            "sha256": hash_seed_file(seed_file_path),
        }

    return {
        "output_dir": output_dir.as_posix(),
        "policy_ids": list(policy_ids),
        "metadata": top_level_metadata,
        "matrices": matrices,
        "posterior_samples": {
            "policy_ids": list(policy_ids),
            "sample_count": sample_count,
            "values": posterior_matrix,
        },
        "matchups": [
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
        ],
    }


def relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = ["build_final_eval_payload", "relative_to"]
