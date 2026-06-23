from __future__ import annotations

from typing import Literal

import pytest
from weiss_rl.eval import (
    EvalGameRecord,
    fold_game_payoff,
    paired_seed_mean_score,
    paired_seed_score,
    paired_seed_scores,
)

_CONFIG_HASH256 = "ab" * 32
_SPEC_HASH256 = "cd" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _pair(
    pair_index: int,
    outcome_a: OutcomeToken,
    outcome_b: OutcomeToken,
    *,
    run_id256: str | None = None,
) -> list[EvalGameRecord]:
    episode_seed = pair_index + 100
    return [
        _record(pair_index, 0, outcome_a, episode_seed=episode_seed, run_id256=run_id256),
        _record(pair_index, 1, outcome_b, episode_seed=episode_seed, run_id256=run_id256),
    ]


def _duplicate_seed_runs(episode_seed: int, *outcomes: tuple[OutcomeToken, OutcomeToken]) -> list[EvalGameRecord]:
    records: list[EvalGameRecord] = []
    for pair_index, (outcome_a, outcome_b) in enumerate(outcomes):
        records.extend(
            [
                _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
                _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
            ]
        )
    return records


def _record(
    pair_index: int,
    swap_index: int,
    outcome: OutcomeToken,
    *,
    episode_seed: int | None = None,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
    run_id256: str | None = None,
) -> EvalGameRecord:
    normalized_swap_index = int(swap_index)
    if normalized_swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1

    episode_seed_value = pair_index if episode_seed is None else episode_seed
    episode_index = pair_index * 2 + normalized_swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=normalized_swap_index,
        episode_index=episode_index,
        episode_seed=episode_seed_value,
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
        run_id256=run_id256,
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("W", {"S0": 1.0, "S1": 1.0, "S2": 1.0}),
        ("L", {"S0": 0.0, "S1": 0.0, "S2": 0.0}),
        ("D", {"S0": 0.5, "S1": 0.5, "S2": 0.5}),
        ("T", {"S0": 0.0, "S1": 0.0, "S2": None}),
    ],
)
def test_fold_game_payoff_matches_s0_s1_s2_rules(
    outcome: OutcomeToken,
    expected: dict[str, float | None],
) -> None:
    assert fold_game_payoff(outcome, scheme="S0") == expected["S0"]
    assert fold_game_payoff(outcome, scheme="S1") == expected["S1"]
    assert fold_game_payoff(outcome, scheme="S2") == expected["S2"]


def test_paired_seed_score_applies_scheme_rules() -> None:
    pair_records = _pair(7, "W", "T")

    assert paired_seed_score(pair_records, scheme="S0") == 0.5
    assert paired_seed_score(list(reversed(pair_records)), scheme="S1") == 0.5
    assert paired_seed_score(pair_records, scheme="S2") == 1.0
    assert paired_seed_score(_pair(8, "T", "T"), scheme="S2") is None


def test_paired_seed_score_aggregates_duplicate_same_seed_runs() -> None:
    records = _duplicate_seed_runs(250, ("W", "L"), ("W", "W"))

    assert paired_seed_score(records, scheme="S0") == pytest.approx(0.75)
    assert paired_seed_score(records, scheme="S2") == pytest.approx(0.75)


def test_paired_seed_mean_score_returns_expected_exact_p_ij_mean() -> None:
    records = [
        *_pair(0, "W", "L"),
        *_pair(1, "W", "T"),
        *_pair(2, "D", "T"),
        *_pair(3, "W", "W"),
    ]

    assert paired_seed_mean_score(records, scheme="S0") == 0.5625
    assert paired_seed_mean_score(records, scheme="S1") == 0.5625
    assert paired_seed_mean_score(records, scheme="S2") == 0.75


def test_paired_seed_mean_score_is_order_independent() -> None:
    records = [
        *_pair(0, "W", "L"),
        *_pair(1, "W", "T"),
        *_pair(2, "D", "T"),
        *_pair(3, "W", "W"),
    ]
    shuffled = [records[5], records[0], records[7], records[2], records[6], records[1], records[4], records[3]]

    assert paired_seed_mean_score(shuffled, scheme="S0") == paired_seed_mean_score(records, scheme="S0")
    assert paired_seed_mean_score(shuffled, scheme="S2") == paired_seed_mean_score(records, scheme="S2")


def test_paired_seed_scores_group_same_pair_index_by_run_identity() -> None:
    records = [
        *_pair(0, "W", "W", run_id256="11" * 32),
        *_pair(0, "L", "L", run_id256="22" * 32),
    ]

    assert paired_seed_scores(records, scheme="S0") == pytest.approx((1.0, 0.0))
    assert paired_seed_mean_score(records, scheme="S0") == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("records", "expected_message"),
    [
        (_pair(0, "W", "L")[:1], "at least 2 records"),
        ([_record(1, 0, "W", episode_seed=10), _record(1, 1, "L", episode_seed=11)], "must share episode_seed"),
        ([_record(2, 0, "W"), _record(2, 0, "L")], "matching counts for swap_index 0 and 1"),
    ],
)
def test_paired_seed_score_rejects_malformed_pair_groups(
    records: list[EvalGameRecord],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        paired_seed_score(records, scheme="S0")


def test_paired_seed_mean_score_s2_rejects_all_excluded_pairs() -> None:
    records = [*_pair(0, "T", "T"), *_pair(1, "T", "T")]

    with pytest.raises(ValueError, match="S2 excluded all paired seeds"):
        paired_seed_mean_score(records, scheme="S2")
