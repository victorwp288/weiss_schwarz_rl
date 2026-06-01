"""Console reporting for paired-outcome overlap reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_overlap_report_output_payload(*, output_json: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "report_count": report["report_count"],
        "total_conflict_key_count": report["total_conflict_key_count"],
        "total_truncated_rows": report["total_truncated_rows"],
    }


def paired_outcome_overlap_report_output_line(*, output_json: Path, report: Mapping[str, Any]) -> str:
    return json.dumps(
        paired_outcome_overlap_report_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_overlap_report_output_line",
    "paired_outcome_overlap_report_output_payload",
]
