"""Final-eval top-level payload assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.analysis.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.final.matrices import (
    build_final_eval_matrices,
    build_final_eval_posterior_samples,
    canonical_matchup_results_by_key,
)
from weiss_rl.eval.final.payload_sections import (
    build_final_eval_matchup_rows,
    build_final_eval_metadata,
    final_eval_summary_sections_payload,
    relative_to,
)


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
    canonical_results = canonical_matchup_results_by_key(matchup_results)

    return {
        "output_dir": output_dir.as_posix(),
        "policy_ids": list(policy_ids),
        "summary_sections": final_eval_summary_sections_payload(),
        "metadata": build_final_eval_metadata(
            metadata=metadata,
            policy_ids=policy_ids,
            matchup_results=matchup_results,
            stage1_paired_seeds=stage1_paired_seeds,
            max_paired_seeds=max_paired_seeds,
            paired_seeds=paired_seeds,
            stop_rules=stop_rules,
            scheme=scheme,
            sample_count=sample_count,
            selection_payload=selection_payload,
            seed_file_path=seed_file_path,
        ),
        "matrices": build_final_eval_matrices(
            policy_ids=policy_ids,
            canonical_results_by_key=canonical_results,
        ),
        "posterior_samples": build_final_eval_posterior_samples(
            policy_ids=policy_ids,
            canonical_results_by_key=canonical_results,
            sample_count=sample_count,
        ),
        "matchups": build_final_eval_matchup_rows(output_dir=output_dir, matchup_results=matchup_results),
    }


__all__ = ["build_final_eval_payload", "final_eval_summary_sections_payload", "relative_to"]
