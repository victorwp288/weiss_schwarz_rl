from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from weiss_rl.eval.harness import GameResult
from weiss_rl.runtime.components.outcomes import (
    MIRROR_OPPONENT_POLICY_ID,
    apply_outcome_counters_to_tracker,
    update_outcomes,
    update_outcomes_from_transition_arrays,
)


class _Tracker:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str]] = []

    def update(self, opponent_id: str, outcome: str) -> None:
        self.updates.append((opponent_id, outcome))


@dataclass(frozen=True)
class _TerminalLookup:
    results: dict[int, GameResult]
    calls: list[tuple[int, int]]

    def __call__(self, _batch: Any, *, env_index: int, acting_seat: int) -> GameResult:
        self.calls.append((int(env_index), int(acting_seat)))
        return self.results[int(env_index)]


def test_update_outcomes_skips_mirror_and_maps_terminal_results() -> None:
    tracker = _Tracker()
    lookup = _TerminalLookup(
        results={
            1: GameResult(episode_seed=1, terminated=True, truncated=False, winner_seat=0),
            2: GameResult(episode_seed=2, terminated=True, truncated=False, winner_seat=None),
            3: GameResult(episode_seed=3, terminated=False, truncated=True, winner_seat=None),
        },
        calls=[],
    )

    update_outcomes(
        outcome_tracker=tracker,
        opponent_policy_id_by_env=np.array(
            [MIRROR_OPPONENT_POLICY_ID, "policy_win", "policy_draw", "policy_timeout"],
            dtype=object,
        ),
        focal_seat_by_env=np.array([0, 0, 1, 1], dtype=np.int64),
        acting_seat=np.array([1, 1, 0, 1], dtype=np.int64),
        terminal_batch=object(),
        done=np.array([True, True, True, True], dtype=np.bool_),
        game_result_from_step_fn=lookup,
    )

    assert lookup.calls == [(1, 1), (2, 0), (3, 1)]
    assert tracker.updates == [
        ("policy_win", "w"),
        ("policy_draw", "d"),
        ("policy_timeout", "t"),
    ]


def test_update_outcomes_from_transition_arrays_maps_reward_perspective() -> None:
    tracker = _Tracker()
    counters: dict[str, int] = {}

    update_outcomes_from_transition_arrays(
        outcome_tracker=tracker,
        opponent_policy_id_by_env=np.array(
            [MIRROR_OPPONENT_POLICY_ID, "policy_win", "policy_loss", "policy_draw", "policy_timeout"],
            dtype=object,
        ),
        focal_seat_by_env=np.array([0, 0, 0, 1, 1], dtype=np.int64),
        acting_seat=np.array([1, 0, 1, 0, 1], dtype=np.int64),
        rewards=np.array([1.0, 1.0, 1.0, 0.0, -1.0], dtype=np.float32),
        truncated=np.array([False, False, False, False, True], dtype=np.bool_),
        done=np.array([True, True, True, True, True], dtype=np.bool_),
        counters=counters,
    )

    assert tracker.updates == [
        ("policy_win", "w"),
        ("policy_loss", "l"),
        ("policy_draw", "d"),
        ("policy_timeout", "t"),
    ]
    replay_tracker = _Tracker()
    applied = apply_outcome_counters_to_tracker(outcome_tracker=replay_tracker, counters=counters)

    assert applied == 4
    assert replay_tracker.updates == tracker.updates


def test_update_outcomes_from_transition_arrays_ignores_incomplete_rows() -> None:
    tracker = _Tracker()

    update_outcomes_from_transition_arrays(
        outcome_tracker=tracker,
        opponent_policy_id_by_env=np.array(["policy_pending"], dtype=object),
        focal_seat_by_env=np.array([0], dtype=np.int64),
        acting_seat=np.array([0], dtype=np.int64),
        rewards=np.array([1.0], dtype=np.float32),
        truncated=np.array([False], dtype=np.bool_),
        done=np.array([False], dtype=np.bool_),
    )

    assert tracker.updates == []
