from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def anchor_scores(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for key, raw_score in value.items():
        score = finite_float(raw_score)
        if score is not None:
            scores[str(key)] = score
    return scores


def load_reference_anchor_scores(path: Path) -> dict[str, float]:
    """Load opponent anchor scores from a targeted-confirm summary or a simple mapping."""

    payload = json_object(path)
    if payload is None:
        raise ValueError(f"reference summary must be a JSON object: {path}")
    direct_scores = anchor_scores(payload.get("anchor_scores"))
    if direct_scores:
        return direct_scores
    rows = payload.get("rows")
    if isinstance(rows, list):
        scores: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = row.get("opponent_policy_id")
            mean = finite_float(row.get("mean"))
            if isinstance(opponent, str) and opponent and mean is not None:
                scores[opponent] = mean
        if scores:
            return scores
    return anchor_scores(payload)


def reference_comparison(
    anchor_scores_by_name: Mapping[str, float],
    *,
    reference_anchor_scores: Mapping[str, float],
    reference_label: str,
) -> dict[str, Any] | None:
    common_anchors = sorted(set(anchor_scores_by_name) & set(reference_anchor_scores))
    if not common_anchors:
        return None
    anchor_deltas = {
        anchor: float(anchor_scores_by_name[anchor]) - float(reference_anchor_scores[anchor])
        for anchor in common_anchors
    }
    deltas = list(anchor_deltas.values())
    return {
        "reference_label": str(reference_label),
        "common_anchors": common_anchors,
        "reference_anchor_scores": {anchor: float(reference_anchor_scores[anchor]) for anchor in common_anchors},
        "anchor_deltas": anchor_deltas,
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "all_common_at_or_above_reference": all(delta >= 0.0 for delta in deltas),
    }
