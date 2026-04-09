from __future__ import annotations

from typing import Literal

import pytest

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import EvalGameRecord, MatchupSummary, summarize_stage2_records
from weiss_rl.eval.uncertainty import EvalUncertaintySummary

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
) -> EvalGameRecord:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1

    episode_index = pair_index * 2 + swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=swap_index,
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


def test_summarize_stage2_records_reports_summary_and_continue_state() -> None:
    decision = summarize_stage2_records(
        [*_pair(0, "W", "L"), *_pair(1, "W", "W"), *_pair(2, "D", "L")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        max_paired_seeds=10,
        sample_count=64,
        seed=7,
    )

    assert decision.summary == MatchupSummary(games=6, wins=3, losses=2, draws=1, truncations=0, engine_errors=0)
    assert decision.uncertainty is not None
    assert decision.uncertainty.paired_seed_count == 3
    assert decision.observed_paired_seeds == 3
    assert decision.excluded_paired_seeds == 0
    assert decision.stop_reason == "continue"
    assert decision.should_stop is False


@pytest.mark.parametrize(
    ("uncertainty", "expected_reason"),
    [
        (
            EvalUncertaintySummary(
                mean=0.8,
                ci_low=0.7,
                ci_high=0.9,
                ci_half_width=0.1,
                prob_gt_half=0.97,
                prob_lt_half=0.0,
                paired_seed_count=2,
                sample_count=16,
            ),
            "decisive",
        ),
        (
            EvalUncertaintySummary(
                mean=0.55,
                ci_low=0.53,
                ci_high=0.57,
                ci_half_width=0.02,
                prob_gt_half=0.6,
                prob_lt_half=0.0,
                paired_seed_count=2,
                sample_count=16,
            ),
            "precision",
        ),
    ],
)
def test_summarize_stage2_records_selects_stop_reason_by_priority(
    monkeypatch: pytest.MonkeyPatch,
    uncertainty: EvalUncertaintySummary,
    expected_reason: str,
) -> None:
    def _fake_uncertainty(*args, **kwargs):
        return uncertainty

    monkeypatch.setattr("weiss_rl.eval.stage2.bayesian_bootstrap_summary", _fake_uncertainty)

    decision = summarize_stage2_records(
        [*_pair(0, "W", "L")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        max_paired_seeds=4,
    )

    assert decision.stop_reason == expected_reason
    assert decision.should_stop is True


def test_summarize_stage2_records_uses_stop_confidence_as_default_ci_level(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

    def _fake_uncertainty(*args, **kwargs):
        captured["ci_level"] = kwargs["ci_level"]
        return EvalUncertaintySummary(
            mean=0.5,
            ci_low=0.4,
            ci_high=0.6,
            ci_half_width=0.1,
            prob_gt_half=0.2,
            prob_lt_half=0.2,
            paired_seed_count=1,
            sample_count=8,
        )

    monkeypatch.setattr("weiss_rl.eval.stage2.bayesian_bootstrap_summary", _fake_uncertainty)

    summarize_stage2_records(
        [*_pair(0, "W", "L")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.9),
        max_paired_seeds=4,
    )

    assert captured["ci_level"] == 0.9


def test_summarize_stage2_records_handles_all_excluded_s2_pairs_before_budget() -> None:
    decision = summarize_stage2_records(
        [*_pair(0, "T", "T")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        max_paired_seeds=2,
        scheme="S2",
    )

    assert decision.uncertainty is None
    assert decision.paired_seed_count == 0
    assert decision.observed_paired_seeds == 1
    assert decision.excluded_paired_seeds == 1
    assert decision.has_payoff_samples is False
    assert decision.stop_reason == "continue"
    assert decision.should_stop is False


def test_summarize_stage2_records_marks_budgeted_all_excluded_s2_pairs_explicitly() -> None:
    decision = summarize_stage2_records(
        [*_pair(0, "T", "T"), *_pair(1, "T", "T")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        max_paired_seeds=2,
        scheme="S2",
    )

    assert decision.uncertainty is None
    assert decision.paired_seed_count == 0
    assert decision.observed_paired_seeds == 2
    assert decision.excluded_paired_seeds == 2
    assert decision.stop_reason == "no_included_pairs"
    assert decision.should_stop is True


def test_summarize_stage2_records_uses_observed_pairs_for_budget_under_s2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_uncertainty(*args, **kwargs):
        return EvalUncertaintySummary(
            mean=0.5,
            ci_low=0.25,
            ci_high=0.75,
            ci_half_width=0.25,
            prob_gt_half=0.4,
            prob_lt_half=0.4,
            paired_seed_count=1,
            sample_count=32,
        )

    monkeypatch.setattr("weiss_rl.eval.stage2.bayesian_bootstrap_summary", _fake_uncertainty)

    decision = summarize_stage2_records(
        [*_pair(0, "W", "L"), *_pair(1, "T", "T")],
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.001, stop_confidence=0.999),
        max_paired_seeds=2,
        scheme="S2",
        sample_count=32,
        seed=7,
    )

    assert decision.uncertainty is not None
    assert decision.paired_seed_count == 1
    assert decision.observed_paired_seeds == 2
    assert decision.excluded_paired_seeds == 1
    assert decision.stop_reason == "budget"
    assert decision.should_stop is True


def test_summarize_stage2_records_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_paired_seeds must be positive"):
        summarize_stage2_records(
            [*_pair(0, "W", "L")],
            stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
            max_paired_seeds=0,
        )
