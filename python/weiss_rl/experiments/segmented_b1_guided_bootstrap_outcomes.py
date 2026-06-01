from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.bootstrap_commands import read_json_object


@dataclass(frozen=True, slots=True)
class SegmentSelectionResult:
    selected: dict[str, Any]
    selection_score: float
    selected_minus_previous: float | None
    latest_minus_best: float | None


@dataclass(frozen=True, slots=True)
class SegmentStopDecision:
    status: str | None
    stop_reason: str | None = None

    @property
    def should_stop(self) -> bool:
        return self.status is not None


def selected_candidate(path: Path) -> dict[str, Any]:
    payload = read_json_object(path)
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"candidate selector did not produce a selected candidate: {path}")
    return selected


def selection_score(candidate: Mapping[str, Any]) -> float:
    raw_score = candidate.get("selection_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return 0.0
    return float(raw_score)


def latest_minus_best(path: Path) -> float | None:
    payload = read_json_object(path)
    run_summaries = payload.get("run_summaries")
    if not isinstance(run_summaries, list) or not run_summaries:
        return None
    raw_value = run_summaries[0].get("latest_minus_best") if isinstance(run_summaries[0], Mapping) else None
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    return float(raw_value)


def read_segment_selection_result(
    final_json: Path,
    *,
    previous_score: float | None,
) -> SegmentSelectionResult:
    selected = selected_candidate(final_json)
    score = selection_score(selected)
    return SegmentSelectionResult(
        selected=selected,
        selection_score=score,
        selected_minus_previous=None if previous_score is None else score - previous_score,
        latest_minus_best=latest_minus_best(final_json),
    )


def populate_completed_segment_record(
    segment_record: dict[str, Any],
    result: SegmentSelectionResult,
) -> None:
    segment_record["status"] = "completed"
    segment_record["selected"] = result.selected
    segment_record["selection_score"] = result.selection_score
    segment_record["selected_minus_previous"] = result.selected_minus_previous
    segment_record["latest_minus_best"] = result.latest_minus_best


def stop_decision(
    result: SegmentSelectionResult,
    *,
    max_selected_drop: float,
    stop_on_latest_falloff: bool,
    max_latest_drop: float,
) -> SegmentStopDecision:
    if not bool(result.selected.get("eligible")):
        return SegmentStopDecision(
            status="stopped_ineligible",
            stop_reason="selected candidate did not meet required anchor threshold",
        )
    if result.selected_minus_previous is not None and result.selected_minus_previous < -float(max_selected_drop):
        return SegmentStopDecision(
            status="stopped_selected_drop",
            stop_reason=(
                f"selected score dropped by {result.selected_minus_previous:.4f}, below -{float(max_selected_drop):.4f}"
            ),
        )
    if (
        bool(stop_on_latest_falloff)
        and result.latest_minus_best is not None
        and result.latest_minus_best < -float(max_latest_drop)
    ):
        return SegmentStopDecision(
            status="stopped_latest_falloff",
            stop_reason=(
                f"latest fell behind best by {result.latest_minus_best:.4f}, below -{float(max_latest_drop):.4f}"
            ),
        )
    return SegmentStopDecision(status=None)
