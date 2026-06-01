"""Console reporting for paired-swing opponent-context margin reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_swing_context_margins_output_payload(*, output_json: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "row_count": report["row_count"],
        "context_episode_count": report["context_episode_count"],
        "missing_context_episode_count": report["context_coverage"]["missing_context_episode_count"],
        "positive_margin_min": report["positive_margin_min"],
        "positive_margin_mean": report["positive_margin_mean"],
    }


def paired_swing_context_margins_output_line(*, output_json: Path, report: Mapping[str, Any]) -> str:
    return json.dumps(
        paired_swing_context_margins_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_swing_context_margins_output_line",
    "paired_swing_context_margins_output_payload",
]
