from __future__ import annotations


def linear_anneal_value(
    *,
    initial_value: float,
    final_value: float,
    start_update: int = 0,
    end_update: int = -1,
    update_count: int,
) -> float:
    """Linearly anneal from ``initial_value`` to ``final_value`` over an update window."""

    initial = float(initial_value)
    final = float(final_value)
    start = max(0, int(start_update))
    end = int(end_update)
    current = max(0, int(update_count))

    if end < 0 or initial == final:
        return initial
    if end <= start:
        return final if current >= end else initial
    if current <= start:
        return initial
    if current >= end:
        return final
    progress = float(current - start) / float(end - start)
    return initial + (final - initial) * progress
