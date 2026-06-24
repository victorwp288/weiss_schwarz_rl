"""Counter updates for learner-side reward shaping penalties."""

from __future__ import annotations


def record_reward_penalty(
    counters: dict[str, int],
    *,
    counter_prefix: str,
    count: int,
    total_micros: int,
) -> None:
    """Accumulate count and reward-micro totals for one shaping rule."""

    counters[f"{counter_prefix}_count"] += int(count)
    counters[f"{counter_prefix}_total_micros"] += int(total_micros)
