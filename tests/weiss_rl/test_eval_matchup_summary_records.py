from __future__ import annotations

from pathlib import Path

from weiss_rl.eval.simulator.harness import (
    GameResult,
    MatchupSummary,
    ScheduledGame,
    record_completed_game,
    run_seat_swapped_matchup,
    summarize_game_records,
    summarize_pair_outcomes,
)

from tests.weiss_rl.eval_harness_test_support import _CONFIG_HASH256, _RUN_ID256, _SPEC_HASH256, _FakeRunner


def test_summarize_game_records_counts_wldt_and_engine_errors(tmp_path: Path) -> None:
    records = run_seat_swapped_matchup(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[1, 2],
        runner=_FakeRunner(
            [
                GameResult(episode_seed=1, terminated=True, truncated=False, winner_seat=0, engine_status=0),
                GameResult(episode_seed=1, terminated=True, truncated=False, winner_seat=0, engine_status=0),
                GameResult(episode_seed=2, terminated=True, truncated=False, winner_seat=None, engine_status=9),
                GameResult(episode_seed=2, terminated=False, truncated=True, winner_seat=None, engine_status=0),
            ]
        ),
        episodes_path=tmp_path / "episodes.jsonl",
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    ).records

    summary = summarize_game_records(records)

    assert summary == MatchupSummary(
        games=4,
        wins=1,
        losses=1,
        draws=1,
        truncations=1,
        engine_errors=1,
        natural_timeouts=1,
        timeout_unknown=1,
    )
    assert summarize_pair_outcomes(["w", "L", "d", "t"]) == MatchupSummary(
        games=4,
        wins=1,
        losses=1,
        draws=1,
        truncations=1,
        engine_errors=0,
    )


def test_summarize_game_records_aggregates_action_diagnostics() -> None:
    records = [
        record_completed_game(
            scheduled_game=ScheduledGame(
                pair_index=0,
                swap_index=0,
                episode_index=0,
                episode_seed=10,
                focal_policy_id="champion",
                opponent_policy_id="baseline",
                seat0_policy_id="champion",
                seat1_policy_id="baseline",
                focal_seat=0,
            ),
            result=GameResult(
                episode_seed=10,
                terminated=True,
                truncated=False,
                winner_seat=0,
                total_actions=8,
                pass_actions=2,
                main_move_actions=3,
                pass_with_nonpass_available=1,
                max_consecutive_main_moves=2,
            ),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
        ),
        record_completed_game(
            scheduled_game=ScheduledGame(
                pair_index=0,
                swap_index=1,
                episode_index=1,
                episode_seed=10,
                focal_policy_id="champion",
                opponent_policy_id="baseline",
                seat0_policy_id="baseline",
                seat1_policy_id="champion",
                focal_seat=1,
            ),
            result=GameResult(
                episode_seed=10,
                terminated=False,
                truncated=True,
                winner_seat=None,
                total_actions=6,
                pass_actions=1,
                main_move_actions=4,
                pass_with_nonpass_available=0,
                max_consecutive_main_moves=4,
            ),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
        ),
    ]

    summary = summarize_game_records(records)

    assert summary.total_actions == 14
    assert summary.pass_actions == 3
    assert summary.main_move_actions == 7
    assert summary.pass_with_nonpass_available == 1
    assert summary.max_consecutive_main_moves == 4


def test_truncation_is_preserved_raw_in_record_and_summary(tmp_path: Path) -> None:
    result = run_seat_swapped_matchup(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[99],
        runner=_FakeRunner(
            [
                GameResult(episode_seed=99, terminated=False, truncated=True, winner_seat=None, engine_status=17),
                GameResult(episode_seed=99, terminated=True, truncated=False, winner_seat=0, engine_status=0),
            ]
        ),
        episodes_path=tmp_path / "episodes.jsonl",
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    )

    first_record = result.records[0]
    assert first_record.outcome == "T"
    assert first_record.truncated is True
    assert first_record.engine_status == 17
    assert result.summary.truncations == 1
    assert result.summary.engine_errors == 1
