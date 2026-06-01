"""Console reporting for paired-swing mechanistic gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_swing_mechanistic_gate_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "passed": bool(report["passed"]),
        "failures": report["failures"],
        "summary": report["summary"],
    }


def paired_swing_mechanistic_gate_output_line(*, output_json: Path, report: Mapping[str, Any]) -> str:
    return json.dumps(
        paired_swing_mechanistic_gate_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_swing_mechanistic_gate_output_line",
    "paired_swing_mechanistic_gate_output_payload",
]
