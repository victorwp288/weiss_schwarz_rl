"""Console reporting for paired targeted-outcome comparison reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_compare_output_payload(*, report: Mapping[str, Any], output_json: Path) -> dict[str, Any]:
    groups = report["groups"]
    return {
        "output_json": output_json.as_posix(),
        "all_delta_wins": groups["all_compared"]["delta_wins"],
        "fixed_delta_wins": groups["fixed_baselines"]["delta_wins"],
        "learned_delta_wins": groups["learned_opponents"]["delta_wins"],
    }


def paired_outcome_compare_output_line(*, report: Mapping[str, Any], output_json: Path) -> str:
    return json.dumps(paired_outcome_compare_output_payload(report=report, output_json=output_json), sort_keys=True)


__all__ = ["paired_outcome_compare_output_line", "paired_outcome_compare_output_payload"]
