"""Console reporting for paired-swing preference conflict reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_swing_conflict_report_output_payload(*, output_json: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "preference_row_count": report["preference_row_count"],
        "current_state_conflict_count": report["current_state_conflict_count"],
        "history_conflict_count": report["history_conflict_count"],
    }


def paired_swing_conflict_report_output_line(*, output_json: Path, report: Mapping[str, Any]) -> str:
    return json.dumps(
        paired_swing_conflict_report_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_swing_conflict_report_output_line",
    "paired_swing_conflict_report_output_payload",
]
