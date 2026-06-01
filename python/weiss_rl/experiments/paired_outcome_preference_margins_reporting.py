"""Console reporting for paired-outcome preference margin reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_preference_margins_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "pair_count": report["pair_count"],
        "train_rows": report["train_rows"],
        "dpo_margin_mean": report["dpo_margin_mean"],
        "dpo_margin_min": report["dpo_margin_min"],
        "satisfied_fraction": report["satisfied_fraction"],
        "current_context_episode_count": report["current_context_episode_count"],
        "reference_context_episode_count": report["reference_context_episode_count"],
        "current_missing_context_episode_count": report["current_context_coverage"]["missing_context_episode_count"],
        "reference_missing_context_episode_count": report["reference_context_coverage"][
            "missing_context_episode_count"
        ],
    }


def paired_outcome_preference_margins_output_line(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_margins_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_margins_output_line",
    "paired_outcome_preference_margins_output_payload",
]
