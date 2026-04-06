from __future__ import annotations

import pytest

from weiss_rl.league.outcomes import OnlineOutcomeTracker


def test_online_outcome_tracker_applies_sliding_window_per_opponent() -> None:
    tracker = OnlineOutcomeTracker(window_size=3)

    for outcome in ["w", "l", "d", "w"]:
        tracker.update("snapshot_a", outcome)
    tracker.update("snapshot_b", "l")

    assert tracker.counts("snapshot_a") == (1, 1, 1)
    assert tracker.win_rate("snapshot_a") == pytest.approx(0.5)
    assert tracker.counts("snapshot_b") == (0, 1, 0)
    assert tracker.win_rate("snapshot_b") == pytest.approx(0.0)
    assert tracker.win_rates(["snapshot_a", "missing", "snapshot_b"]) == pytest.approx([0.5, 0.5, 0.0])


def test_online_outcome_tracker_normalizes_tokens_and_rejects_empty_opponent_ids() -> None:
    tracker = OnlineOutcomeTracker(window_size=2)

    tracker.update("snapshot_a", "W")
    tracker.update("snapshot_a", "d")

    assert tracker.counts("snapshot_a") == (1, 0, 1)
    assert tracker.win_rate("snapshot_a") == pytest.approx(0.75)

    with pytest.raises(ValueError, match="opponent_id must be non-empty"):
        tracker.update("   ", "w")
