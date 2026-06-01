"""Console reporting for paired-outcome preference span audits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def paired_outcome_preference_span_audit_output_payload(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "output_json": output_json.as_posix(),
        "passed": report["span_gate"]["passed"],
        "complete_pair_count": report["complete_pair_count"],
        "different_action_count": report["different_action_count"],
        "compact_span_count": report["compact_span_count"],
        "passing_opponents": report["span_gate"]["passing_opponents"],
    }


def paired_outcome_preference_span_audit_output_line(
    *,
    output_json: Path,
    report: Mapping[str, Any],
) -> str:
    return json.dumps(
        paired_outcome_preference_span_audit_output_payload(output_json=output_json, report=report),
        sort_keys=True,
    )


__all__ = [
    "paired_outcome_preference_span_audit_output_line",
    "paired_outcome_preference_span_audit_output_payload",
]
