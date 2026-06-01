"""Console reporting for paired-swing repair-target reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_swing_report_output_payload(*, output_json: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"]["groups"]
    return {
        "output_json": output_json.as_posix(),
        "all_delta_wins": aggregate["all"]["delta_wins"],
        "fixed_delta_wins": aggregate["fixed"]["delta_wins"],
        "learned_delta_wins": aggregate["learned"]["delta_wins"],
        "hard_negative_delta_wins": aggregate["hard_negative"]["delta_wins"],
    }


def paired_swing_report_output_line(*, output_json: Path, report: Mapping[str, Any]) -> str:
    return json.dumps(paired_swing_report_output_payload(output_json=output_json, report=report), sort_keys=True)


__all__ = [
    "paired_swing_report_output_line",
    "paired_swing_report_output_payload",
]
