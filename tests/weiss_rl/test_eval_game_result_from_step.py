from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.eval.harness import GameResult, game_result_from_step


def test_game_result_from_step_uses_reward_perspective_seat() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0, -1.0], dtype=np.float32),
        actor=np.array([0, 1], dtype=np.int8),
        terminated=np.array([True, True]),
        truncated=np.array([False, False]),
        engine_status=np.array([0, 3], dtype=np.uint8),
        episode_seed=np.array([10, 11], dtype=np.uint64),
        episode_key=np.array([100, 200], dtype=np.uint64),
    )

    record_a = game_result_from_step(step, env_index=0)
    record_b = game_result_from_step(step, env_index=1)

    assert record_a == GameResult(
        episode_seed=10,
        terminated=True,
        truncated=False,
        winner_seat=0,
        engine_status=0,
        termination_reason="terminated",
        simulator_episode_key=100,
    )
    assert record_b == GameResult(
        episode_seed=11,
        terminated=True,
        truncated=False,
        winner_seat=0,
        engine_status=3,
        termination_reason="engine_fault",
        simulator_episode_key=200,
    )


def test_game_result_from_step_prefers_explicit_winner_seat_metadata() -> None:
    step = SimpleNamespace(
        reward=np.array([0.25], dtype=np.float32),
        winner_seat=np.array([1], dtype=np.int8),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
        episode_key=np.array([777], dtype=np.uint64),
    )

    assert game_result_from_step(step) == GameResult(
        episode_seed=10,
        terminated=True,
        truncated=False,
        winner_seat=1,
        engine_status=0,
        termination_reason="terminated",
        simulator_episode_key=777,
    )


def test_game_result_from_step_treats_zero_terminal_reward_as_draw_fallback() -> None:
    step = SimpleNamespace(
        reward=np.array([0.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
        episode_key=np.array([555], dtype=np.uint64),
    )

    assert game_result_from_step(step) == GameResult(
        episode_seed=10,
        terminated=True,
        truncated=False,
        winner_seat=None,
        engine_status=0,
        termination_reason="terminated",
        simulator_episode_key=555,
    )


def test_game_result_from_step_uses_pre_step_acting_seat_when_terminal_row_clears_actor() -> None:
    step = SimpleNamespace(
        reward=np.array([-1.0], dtype=np.float32),
        actor=np.array([-1], dtype=np.int8),
        to_play_seat=np.array([-1], dtype=np.int8),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
        episode_key=np.array([999], dtype=np.uint64),
    )

    assert game_result_from_step(step, acting_seat=1) == GameResult(
        episode_seed=10,
        terminated=True,
        truncated=False,
        winner_seat=0,
        engine_status=0,
        termination_reason="terminated",
        simulator_episode_key=999,
    )


def test_game_result_from_step_accepts_caller_supplied_missing_terminal_context() -> None:
    step = SimpleNamespace(
        reward=-1.0,
        terminated=True,
        truncated=False,
        engine_status=0,
        episode_key=999,
    )

    assert game_result_from_step(step, acting_seat=1, episode_seed=10) == GameResult(
        episode_seed=10,
        terminated=True,
        truncated=False,
        winner_seat=0,
        engine_status=0,
        termination_reason="terminated",
        simulator_episode_key=999,
    )


def test_game_result_from_step_rejects_conflicting_caller_supplied_episode_seed() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0], dtype=np.float32),
        actor=np.array([0], dtype=np.int8),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
    )

    with pytest.raises(ValueError, match="episode_seed mismatch"):
        game_result_from_step(step, episode_seed=11)


def test_game_result_from_step_rejects_decisive_terminal_reward_without_seat() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
    )

    with pytest.raises(AttributeError, match="decisive terminated step must expose"):
        game_result_from_step(step)


def test_game_result_from_step_rejects_mismatched_seat_aliases() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0], dtype=np.float32),
        actor=np.array([0], dtype=np.int8),
        to_play_seat=np.array([1], dtype=np.int8),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
    )

    with pytest.raises(ValueError, match="reward perspective seat mismatch"):
        game_result_from_step(step)


def test_game_result_from_step_rejects_pre_step_acting_seat_that_conflicts_with_valid_terminal_alias() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0], dtype=np.float32),
        actor=np.array([0], dtype=np.int8),
        terminated=np.array([True]),
        truncated=np.array([False]),
        engine_status=np.array([0], dtype=np.uint8),
        episode_seed=np.array([10], dtype=np.uint64),
    )

    with pytest.raises(ValueError, match="reward perspective seat mismatch"):
        game_result_from_step(step, acting_seat=1)
