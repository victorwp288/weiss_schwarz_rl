"""Console reporting for paired-outcome preference row guards."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_preference_row_guard_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "passed": report["passed"],
        "failures": report["failures"],
        "row_count": report["row_count"],
        "current_context_episode_count": report["current_context_episode_count"],
        "reference_context_episode_count": report["reference_context_episode_count"],
        "groups": [
            {
                "label": group["label"],
                "protected": group["protected"],
                "required": group["required"],
                "row_count": group["row_count"],
                "mean_target_logp_delta": group["mean_target_logp_delta"],
                "row_worsened_fraction": group["row_worsened_fraction"],
                "rank_worsened_fraction": group["rank_worsened_fraction"],
                "top_family_changed_rate": group["top_family_changed_rate"],
                "lost_target_non_near_tie_rate": group["lost_target_non_near_tie_rate"],
            }
            for group in report["groups"]
        ],
    }


def paired_outcome_preference_row_guard_output_line(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_row_guard_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_row_guard_output_line",
    "paired_outcome_preference_row_guard_output_payload",
]
