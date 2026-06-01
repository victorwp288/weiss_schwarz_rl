"""Console reporting for paired-outcome preference surface-prototype diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_preference_surface_prototype_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "key_mode": report["key_mode"],
        "opponent_key_mode": report["opponent_key_mode"],
        "prototype_train_rows": report["prototype"]["train_rows"],
        "prototype_unique_key_count": report["prototype"]["unique_key_count"],
        "prototype_conflicting_key_count": report["prototype"]["conflicting_key_count"],
        "probes": [
            {
                "label": probe["label"],
                "train_rows": probe["train_rows"],
                "matched_train_rows": probe["matched_train_rows"],
                "matched_rate": probe["matched_rate"],
                "unexpected_matched_rows": probe["unexpected_matched_rows"],
                "conflicting_matched_key_count": probe["conflicting_matched_key_count"],
            }
            for probe in report["probes"]
        ],
    }


def paired_outcome_preference_surface_prototype_output_line(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_surface_prototype_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_surface_prototype_output_line",
    "paired_outcome_preference_surface_prototype_output_payload",
]
