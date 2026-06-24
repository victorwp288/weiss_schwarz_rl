from __future__ import annotations

import pytest
from weiss_rl.artifacts.reproducibility import key256_to_short64
from weiss_rl.eval.simulator.harness import (
    GameResult,
    ScheduledGame,
    record_completed_game,
    resolve_eval_episode_key,
)

from tests.weiss_rl.eval_harness_test_support import _CONFIG_HASH256, _RUN_ID256, _SPEC_HASH256


def test_record_completed_game_stores_required_reproducibility_fields() -> None:
    scheduled_game = ScheduledGame(
        pair_index=2,
        swap_index=1,
        episode_index=5,
        episode_seed=77,
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        seat0_policy_id="baseline",
        seat1_policy_id="champion",
        focal_seat=1,
    )

    result = GameResult(
        episode_seed=77,
        terminated=True,
        truncated=False,
        winner_seat=1,
        simulator_episode_key=1234,
    )
    record = record_completed_game(
        scheduled_game=scheduled_game,
        result=result,
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    )

    expected_episode_key = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=result,
        run_id256=_RUN_ID256,
    )
    assert record.episode_key == expected_episode_key
    assert record.episode_key64 == key256_to_short64(bytes.fromhex(expected_episode_key))
    assert record.config_hash256 == _CONFIG_HASH256
    assert record.spec_hash256 == _SPEC_HASH256
    assert record.run_id256 == _RUN_ID256


def test_record_completed_game_preserves_action_diagnostics() -> None:
    scheduled_game = ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=77,
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        seat0_policy_id="champion",
        seat1_policy_id="baseline",
        focal_seat=0,
    )
    result = GameResult(
        episode_seed=77,
        terminated=True,
        truncated=False,
        winner_seat=0,
        total_actions=19,
        pass_actions=5,
        main_move_actions=7,
        pass_with_nonpass_available=3,
        max_consecutive_main_moves=4,
    )

    record = record_completed_game(
        scheduled_game=scheduled_game,
        result=result,
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    )

    assert record.total_actions == 19
    assert record.pass_actions == 5
    assert record.main_move_actions == 7
    assert record.pass_with_nonpass_available == 3
    assert record.max_consecutive_main_moves == 4


@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (
            GameResult(episode_seed=7, terminated=False, truncated=False, winner_seat=None),
            "exactly one of terminated or truncated",
        ),
        (
            GameResult(episode_seed=7, terminated=True, truncated=True, winner_seat=None),
            "exactly one of terminated or truncated",
        ),
        (
            GameResult(episode_seed=7, terminated=False, truncated=True, winner_seat=0),
            "cannot include winner_seat",
        ),
    ],
)
def test_record_completed_game_rejects_invalid_terminal_states(result: GameResult, expected_message: str) -> None:
    scheduled_game = ScheduledGame(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=7,
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        seat0_policy_id="champion",
        seat1_policy_id="baseline",
        focal_seat=0,
    )

    with pytest.raises(ValueError, match=expected_message):
        record_completed_game(
            scheduled_game=scheduled_game,
            result=result,
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
        )
