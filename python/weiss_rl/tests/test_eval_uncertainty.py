from __future__ import annotations

from typing import Literal

import pytest

from weiss_rl.eval import (
    EvalGameRecord,
    bayesian_bootstrap_summary,
    paired_seed_scores,
    paired_seed_uncertainty_summary,
)

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
    episode_seed: int | None = None,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
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
    )


def test_paired_seed_scores_are_sorted_by_pair_index() -> None:
    records = [
        *_pair(2, "D", "L"),
        *_pair(0, "W", "L"),
        *_pair(1, "W", "W"),
    ]
    shuffled = [records[1], records[4], records[5], records[0], records[3], records[2]]

    assert paired_seed_scores(shuffled, scheme="S0") == (0.5, 1.0, 0.25)


def test_paired_seed_scores_split_reused_pair_index_by_episode_seed() -> None:
    records = [
        *_pair(0, "W", "L"),
        _record(0, 0, "W", episode_seed=250),
        _record(0, 1, "W", episode_seed=250),
    ]

    assert paired_seed_scores(records, scheme="S0") == (0.5, 1.0)


def test_bayesian_bootstrap_summary_reports_exact_arithmetic_mean() -> None:
    summary = bayesian_bootstrap_summary([0.0, 1.0], sample_count=1, seed=7)

    assert summary.mean == 0.5


def test_bayesian_bootstrap_summary_reports_ci_and_half_width() -> None:
    summary = bayesian_bootstrap_summary([0.25, 0.5, 1.0], sample_count=8, ci_level=0.8, seed=123)

    assert summary.mean == pytest.approx(0.5833333333333334)
    assert summary.ci_low == pytest.approx(0.40789420874755616)
    assert summary.ci_high == pytest.approx(0.709689696309074)
    assert summary.ci_half_width == pytest.approx(0.1508977437807589)
    assert summary.prob_gt_half == pytest.approx(0.375)
    assert summary.prob_lt_half == pytest.approx(0.625)


def test_bayesian_bootstrap_summary_uses_strict_decisive_comparisons() -> None:
    summary = bayesian_bootstrap_summary([0.5, 0.5, 0.5], sample_count=32, seed=9)

    assert summary.prob_gt_half == 0.0
    assert summary.prob_lt_half == 0.0
    assert summary.ci_low == 0.5
    assert summary.ci_high == 0.5
    assert summary.ci_half_width == 0.0


def test_paired_seed_uncertainty_summary_excludes_s2_truncated_pairs() -> None:
    records = [
        *_pair(0, "W", "T"),
        *_pair(1, "T", "T"),
        *_pair(2, "D", "L"),
    ]

    summary = paired_seed_uncertainty_summary(records, scheme="S2", sample_count=16, seed=5)

    assert summary.paired_seed_count == 2
    assert summary.mean == 0.625


def test_paired_seed_uncertainty_summary_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="paired_seed_scores requires at least one record"):
        paired_seed_uncertainty_summary([], scheme="S0")


def test_paired_seed_uncertainty_summary_rejects_all_excluded_s2_pairs() -> None:
    records = [*_pair(0, "T", "T"), *_pair(1, "T", "T")]

    with pytest.raises(ValueError, match="S2 excluded all paired seeds"):
        paired_seed_uncertainty_summary(records, scheme="S2")


def test_paired_seed_uncertainty_summary_is_seed_deterministic_and_order_independent() -> None:
    records = [
        *_pair(0, "W", "L"),
        *_pair(1, "W", "T"),
        *_pair(2, "D", "T"),
        *_pair(3, "W", "W"),
    ]
    shuffled = [records[5], records[0], records[7], records[2], records[6], records[1], records[4], records[3]]

    summary_a = paired_seed_uncertainty_summary(records, scheme="S0", sample_count=32, seed=11)
    summary_b = paired_seed_uncertainty_summary(shuffled, scheme="S0", sample_count=32, seed=11)
    summary_c = paired_seed_uncertainty_summary(records, scheme="S0", sample_count=32, seed=12)

    assert summary_a == summary_b
    assert summary_a != summary_c
