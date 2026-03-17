from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from weiss_rl.eval import EvalGameRecord, build_seat_advantage_diagnostics, write_matchup_diagnostics_json

_CONFIG_HASH256 = "ab" * 32
_SPEC_HASH256 = "cd" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _pair(pair_index: int, outcome_a: OutcomeToken, outcome_b: OutcomeToken) -> list[EvalGameRecord]:
    episode_seed = pair_index + 100
    return [
        _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
        _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
    ]


def _record(
    pair_index: int,
    swap_index: int,
    outcome: OutcomeToken,
    *,
    episode_seed: int,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
    focal_seat: int | None = None,
) -> EvalGameRecord:
    normalized_swap_index = int(swap_index)
    if focal_seat is None:
        focal_seat = normalized_swap_index
    if normalized_swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id

    episode_index = pair_index * 2 + normalized_swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=normalized_swap_index,
        episode_index=episode_index,
        episode_seed=episode_seed,
        episode_key=f"{episode_key64:064x}",
        episode_key64=episode_key64,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome=outcome,
        terminated=outcome != "T",
        truncated=outcome == "T",
        engine_status=0,
    )


def test_build_seat_advantage_diagnostics_tracks_actual_winning_seat_and_both_policies(tmp_path: Path) -> None:
    payload = build_seat_advantage_diagnostics([*_pair(0, "W", "W"), *_pair(1, "L", "T")])

    assert payload["seat_results"] == {
        "seat0_wins": 1,
        "seat1_wins": 2,
        "draws": 0,
        "truncations": 1,
        "engine_errors": 0,
        "decisive_games": 3,
        "total_games": 4,
        "seat0_win_rate": pytest.approx(1 / 3),
        "seat1_win_rate": pytest.approx(2 / 3),
    }
    assert payload["policy_breakdown"] == {
        "champion": {
            "games_as_seat0": 2,
            "games_as_seat1": 2,
            "wins_as_seat0": 1,
            "wins_as_seat1": 1,
            "total_wins": 2,
        },
        "baseline": {
            "games_as_seat0": 2,
            "games_as_seat1": 2,
            "wins_as_seat0": 0,
            "wins_as_seat1": 1,
            "total_wins": 1,
        },
    }

    path = tmp_path / "diagnostics.json"
    write_matchup_diagnostics_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8"))["seat_results"]["seat1_wins"] == 2


def test_build_seat_advantage_diagnostics_rejects_non_swapped_pairs() -> None:
    with pytest.raises(ValueError, match="swap"):
        build_seat_advantage_diagnostics(
            [
                _record(0, 0, "W", episode_seed=100, focal_seat=0),
                _record(0, 1, "L", episode_seed=100, focal_seat=0),
            ]
        )
