from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.eval.harness import (
    GameResult,
    MatchupSummary,
    ScheduledGame,
    abort_on_engine_fault_eval,
    build_seat_swapped_schedule,
    game_result_from_step,
    record_completed_game,
    resolve_eval_episode_key,
    run_seat_swapped_matchup,
    summarize_game_records,
    summarize_pair_outcomes,
)
from weiss_rl.repro import key256_to_hex, key256_to_short64, resolve_episode_key256, stable_hash64

_RUN_ID256 = "ab" * 32
_CONFIG_HASH256 = "cd" * 32
_SPEC_HASH256 = "ef" * 32


class _FakeRunner:
    def __init__(self, results: list[GameResult]) -> None:
        self._results = list(results)
        self.calls: list[ScheduledGame] = []

    def run_game(self, scheduled_game: ScheduledGame) -> GameResult:
        self.calls.append(scheduled_game)
        if not self._results:
            raise AssertionError("fake runner exhausted")
        return self._results.pop(0)


def test_abort_on_engine_fault_eval_writes_artifact_and_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="engine_status!=0 during evaluation"):
        abort_on_engine_fault_eval(
            run_dir=tmp_path,
            engine_status=np.array([0, 9, 4], dtype=np.int32),
            decision_id=np.array([100, 101, 102], dtype=np.int64),
            episode_key=b"episode-7",
        )

    payload = json.loads((tmp_path / "eval_engine_fault.json").read_text(encoding="utf-8"))
    assert payload == {
        "decision_id": [100, 101, 102],
        "engine_status": [0, 9, 4],
        "episode_key": "b'episode-7'",
        "fault_env_indices": [1, 2],
        "note": "engine_status!=0 during evaluation",
    }


def test_abort_on_engine_fault_eval_is_a_noop_without_faults(tmp_path: Path) -> None:
    abort_on_engine_fault_eval(
        run_dir=tmp_path,
        engine_status=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([1, 2], dtype=np.int64),
    )

    assert not (tmp_path / "eval_engine_fault.json").exists()


def test_build_seat_swapped_schedule_uses_fixed_seed_pair_order() -> None:
    schedule = build_seat_swapped_schedule(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[11, 22],
    )

    assert [(game.pair_index, game.swap_index, game.episode_seed) for game in schedule] == [
        (0, 0, 11),
        (0, 1, 11),
        (1, 0, 22),
        (1, 1, 22),
    ]
    assert [(game.seat0_policy_id, game.seat1_policy_id, game.focal_seat) for game in schedule] == [
        ("champion", "baseline", 0),
        ("baseline", "champion", 1),
        ("champion", "baseline", 0),
        ("baseline", "champion", 1),
    ]


def test_run_seat_swapped_matchup_emits_per_game_records_jsonl(tmp_path: Path) -> None:
    runner = _FakeRunner(
        [
            GameResult(episode_seed=7, terminated=True, truncated=False, winner_seat=0, engine_status=0),
            GameResult(episode_seed=7, terminated=True, truncated=False, winner_seat=None, engine_status=5),
        ]
    )

    result = run_seat_swapped_matchup(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[7],
        runner=runner,
        episodes_path=tmp_path / "episodes.jsonl",
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    )

    assert len(result.records) == 2
    assert result.episodes_path == tmp_path / "episodes.jsonl"
    assert [record.outcome for record in result.records] == ["W", "D"]
    assert [record.engine_status for record in result.records] == [0, 5]
    assert [record.truncated for record in result.records] == [False, False]
    assert [record.config_hash256 for record in result.records] == [_CONFIG_HASH256, _CONFIG_HASH256]
    assert [record.spec_hash256 for record in result.records] == [_SPEC_HASH256, _SPEC_HASH256]
    assert [record.episode_key64 for record in result.records] == [
        key256_to_short64(bytes.fromhex(record.episode_key)) for record in result.records
    ]

    payloads = [json.loads(line) for line in result.episodes_path.read_text(encoding="utf-8").splitlines()]
    assert [payload["swap_index"] for payload in payloads] == [0, 1]
    assert [payload["seat0_policy_id"] for payload in payloads] == ["champion", "baseline"]
    assert [payload["seat1_policy_id"] for payload in payloads] == ["baseline", "champion"]
    assert [payload["outcome"] for payload in payloads] == ["W", "D"]
    assert [payload["engine_status"] for payload in payloads] == [0, 5]
    assert [payload["config_hash256"] for payload in payloads] == [_CONFIG_HASH256, _CONFIG_HASH256]
    assert [payload["spec_hash256"] for payload in payloads] == [_SPEC_HASH256, _SPEC_HASH256]
    assert [payload["episode_key64"] for payload in payloads] == [record.episode_key64 for record in result.records]
    assert [call.swap_index for call in runner.calls] == [0, 1]


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

    assert summary == MatchupSummary(games=4, wins=1, losses=1, draws=1, truncations=1, engine_errors=1)
    assert summarize_pair_outcomes(["w", "L", "d", "t"]) == MatchupSummary(
        games=4,
        wins=1,
        losses=1,
        draws=1,
        truncations=1,
        engine_errors=0,
    )


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


def test_episode_key_prefers_simulator_value_and_falls_back_deterministically() -> None:
    scheduled_game = ScheduledGame(
        pair_index=0,
        swap_index=1,
        episode_index=3,
        episode_seed=123,
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        seat0_policy_id="baseline",
        seat1_policy_id="champion",
        focal_seat=1,
    )

    with_simulator_key = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=777,
        ),
        run_id256=_RUN_ID256,
    )
    expected = key256_to_hex(
        resolve_episode_key256(
            simulator_episode_key=777,
            run_id256=bytes.fromhex(_RUN_ID256),
            actor_id=stable_hash64(b"champion") & ((1 << 32) - 1),
            env_id=stable_hash64("champion\0baseline".encode("utf-8")) & ((1 << 32) - 1),
            episode_index=3,
            episode_seed64=123,
        )
    )
    assert with_simulator_key == expected

    without_simulator_key_a = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=None,
        ),
        run_id256=_RUN_ID256,
    )
    without_simulator_key_b = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=None,
        ),
        run_id256=_RUN_ID256,
    )
    assert without_simulator_key_a == without_simulator_key_b
    assert without_simulator_key_a != with_simulator_key


def test_game_result_from_step_normalizes_reward_and_episode_fields() -> None:
    step = SimpleNamespace(
        reward=np.array([1.0, -1.0], dtype=np.float32),
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
        simulator_episode_key=100,
    )
    assert record_b == GameResult(
        episode_seed=11,
        terminated=True,
        truncated=False,
        winner_seat=1,
        engine_status=3,
        simulator_episode_key=200,
    )
