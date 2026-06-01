from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SegmentStopOutcome:
    segment_status: str
    summary_status: str
    stop_reason: str


def rejected_segment_outcome(
    *,
    selected: dict[str, Any],
    guard: dict[str, Any],
    multiobjective_guard: dict[str, Any] | None,
) -> SegmentStopOutcome | None:
    if not bool(selected.get("eligible")):
        return SegmentStopOutcome(
            segment_status="rejected",
            summary_status="stopped_ineligible",
            stop_reason="selected checkpoint did not meet required anchor threshold",
        )
    if not bool(guard["passed"]):
        return SegmentStopOutcome(
            segment_status="rejected",
            summary_status="stopped_guard_failed",
            stop_reason="selected checkpoint failed B2/B3/B4 guard",
        )
    if multiobjective_guard is not None and not bool(multiobjective_guard["passed"]):
        return SegmentStopOutcome(
            segment_status="rejected",
            summary_status="stopped_multiobjective_guard_failed",
            stop_reason="selected checkpoint failed fixed/learned multi-objective guard",
        )
    return None


def publish_confirmation_skip_payload(
    *,
    confirm_paired_seeds: int,
    publish_min_confirm_paired_seeds: int,
    continue_unpublished_confirmed: bool,
) -> dict[str, int | bool | str]:
    return {
        "reason": "confirmation_seed_count_below_publish_minimum",
        "confirm_paired_seeds": int(confirm_paired_seeds),
        "publish_min_confirm_paired_seeds": int(publish_min_confirm_paired_seeds),
        "continued_without_publish": bool(continue_unpublished_confirmed),
    }


def unpublished_confirmation_stop_outcome(
    *,
    continue_unpublished_confirmed: bool,
    has_more_segments: bool,
) -> SegmentStopOutcome | None:
    if bool(continue_unpublished_confirmed):
        if bool(has_more_segments):
            return None
        return SegmentStopOutcome(
            segment_status="accepted_unpublished",
            summary_status="completed_unpublished_confirmation_insufficient",
            stop_reason=(
                "all requested segments passed guard but were not published because confirmation seed count "
                "is below publish_min_confirm_paired_seeds"
            ),
        )
    return SegmentStopOutcome(
        segment_status="accepted_unpublished",
        summary_status="stopped_publish_confirmation_insufficient",
        stop_reason=(
            "selected checkpoint passed guard but was not published because confirmation seed count "
            "is below publish_min_confirm_paired_seeds"
        ),
    )


def apply_segment_stop_outcome(
    *,
    segment_record: MutableMapping[str, Any],
    summary: MutableMapping[str, Any],
    outcome: SegmentStopOutcome,
) -> None:
    segment_record["status"] = outcome.segment_status
    summary["status"] = outcome.summary_status
    summary["stop_reason"] = outcome.stop_reason


__all__ = [
    "SegmentStopOutcome",
    "apply_segment_stop_outcome",
    "publish_confirmation_skip_payload",
    "rejected_segment_outcome",
    "unpublished_confirmation_stop_outcome",
]
