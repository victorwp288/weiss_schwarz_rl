from __future__ import annotations

from typing import Literal

EpisodeEndReason = Literal[
    "terminated",
    "engine_fault",
    "no_progress_timeout",
    "decision_limit_timeout",
    "tick_limit_timeout",
    "timeout_unknown",
]


def classify_episode_end_reason(
    *,
    terminated: bool,
    truncated: bool,
    engine_status: int,
    decision_count: int,
    tick_count: int,
    no_progress_count: int = 0,
    max_decisions: int | None = None,
    max_ticks: int | None = None,
    max_no_progress_decisions: int | None = None,
) -> EpisodeEndReason:
    if int(engine_status) != 0:
        return "engine_fault"
    if not bool(truncated):
        return "terminated"

    decision_count_i = int(decision_count)
    tick_count_i = int(tick_count)
    no_progress_count_i = int(no_progress_count)
    max_decisions_i = None if max_decisions is None else int(max_decisions)
    max_ticks_i = None if max_ticks is None else int(max_ticks)
    max_no_progress_i = None if max_no_progress_decisions is None else int(max_no_progress_decisions)

    if max_decisions_i is not None and decision_count_i >= max_decisions_i:
        return "decision_limit_timeout"
    if max_ticks_i is not None and tick_count_i >= max_ticks_i:
        return "tick_limit_timeout"
    if max_no_progress_i is not None and max_no_progress_i > 0 and no_progress_count_i >= max_no_progress_i:
        return "no_progress_timeout"
    return "timeout_unknown"


def is_natural_timeout_reason(reason: str) -> bool:
    return reason in {"decision_limit_timeout", "tick_limit_timeout", "timeout_unknown"}
