"""Matchup manifest rows for final evaluation artifact exports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.eval.final.payload_sections import relative_to


def final_eval_matchup_manifest_rows(
    *,
    output_dir: Path,
    matchup_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "focal_policy_id": result["focal_policy_id"],
            "opponent_policy_id": result["opponent_policy_id"],
            "matchup_dir": relative_to(Path(result["matchup_dir"]), root=output_dir),
            "paired_seed_count": result["summary"]["paired_seeds"],
            "observed_paired_seed_count": result["summary"]["observed_paired_seeds"],
            "excluded_paired_seed_count": result["summary"]["excluded_paired_seeds"],
            "has_payoff_samples": result["summary"]["has_payoff_samples"],
            "stop_reason": result["summary"]["stop_reason"],
        }
        for result in matchup_results
    ]


__all__ = ["final_eval_matchup_manifest_rows"]
