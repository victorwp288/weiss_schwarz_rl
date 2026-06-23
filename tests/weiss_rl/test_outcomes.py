from __future__ import annotations

import pytest
from weiss_rl.league.outcomes import OnlineOutcomeTracker


def test_online_outcome_tracker_applies_sliding_window_per_opponent() -> None:
    tracker = OnlineOutcomeTracker(window_size=3)

    for outcome in ["w", "l", "d", "w"]:
        tracker.update("snapshot_a", outcome)
    tracker.update("snapshot_b", "l")
    tracker.update("snapshot_b", "t")

    assert tracker.counts("snapshot_a") == (1, 1, 1, 0)
    assert tracker.win_rate("snapshot_a") == pytest.approx(0.5)
    assert tracker.counts("snapshot_b") == (0, 1, 0, 1)
    assert tracker.win_rate("snapshot_b") == pytest.approx(0.0)
    assert tracker.win_rates(["snapshot_a", "missing", "snapshot_b"]) == pytest.approx([0.5, 0.5, 0.0])


def test_online_outcome_tracker_normalizes_tokens_and_rejects_empty_opponent_ids() -> None:
    tracker = OnlineOutcomeTracker(window_size=2)

    tracker.update("snapshot_a", "W")
    tracker.update("snapshot_a", "d")
    tracker.update("snapshot_a", "t")

    assert tracker.counts("snapshot_a") == (0, 0, 1, 1)
    assert tracker.win_rate("snapshot_a") == pytest.approx(0.25)

    with pytest.raises(ValueError, match="opponent_id must be non-empty"):
        tracker.update("   ", "w")


def test_online_outcome_tracker_scopes_counts_by_epoch() -> None:
    tracker = OnlineOutcomeTracker(window_size=4)
    tracker.update("snapshot_a", "w")
    tracker.update("snapshot_a", "l")

    assert tracker.counts("snapshot_a") == (1, 1, 0, 0)

    tracker.bump_epoch(drop_previous=False)
    assert tracker.counts("snapshot_a") == (0, 0, 0, 0)
    tracker.update("snapshot_a", "w")
    tracker.update("snapshot_a", "w")

    assert tracker.counts("snapshot_a") == (2, 0, 0, 0)
    assert tracker.counts("snapshot_a", epoch=0) == (1, 1, 0, 0)
