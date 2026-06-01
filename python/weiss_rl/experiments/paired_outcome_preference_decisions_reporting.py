"""Console reporting for paired-outcome preference decision reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_preference_decisions_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "preference_pair_count": report["preference_pair_count"],
        "complete_pair_count": report["complete_pair_count"],
        "aligned_different_action_count": report["aligned_different_action_count"],
        "same_current_state_edge_count": report["same_current_state_edge_count"],
        "same_current_state_different_action_edge_count": report["same_current_state_different_action_edge_count"],
        "current_state_conflict_count": report["current_state_conflict_count"],
        "history_conflict_count": report["history_conflict_count"],
    }


def paired_outcome_preference_decisions_output_line(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_decisions_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_decisions_output_line",
    "paired_outcome_preference_decisions_output_payload",
]
