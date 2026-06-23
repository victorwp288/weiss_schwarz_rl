from __future__ import annotations

from weiss_rl.config import load_stack_config
from weiss_rl.league import PromotionGateRate
from weiss_rl.league.promotion_gate import _decision_reasons

from ._config_paths import canonical_stack_config_path
from .promotion_gate_test_support import anchor_result, posterior


def test_decision_reasons_use_strict_overall_and_anchor_thresholds() -> None:
    stack = load_stack_config(canonical_stack_config_path())
    anchors = (anchor_result("B0 RandomLegal", prob_lt_guardrail=0.04),)
    truncation = PromotionGateRate(numerator=0, denominator=20, rate=0.0)

    overall_boundary = _decision_reasons(
        anchor_results=anchors,
        overall=posterior(prob_gt_target=0.95),
        truncation=truncation,
        stack=stack,
    )
    assert {reason["code"] for reason in overall_boundary} == {"overall_posterior_below_threshold"}

    anchor_boundary = _decision_reasons(
        anchor_results=(anchor_result("B0 RandomLegal", prob_lt_guardrail=0.05),),
        overall=posterior(prob_gt_target=0.950001),
        truncation=truncation,
        stack=stack,
    )
    assert {reason["code"] for reason in anchor_boundary} == {"anchor_loss_guardrail_exceeded"}


def test_decision_reasons_allow_truncation_limit_and_reject_above() -> None:
    stack = load_stack_config(canonical_stack_config_path())
    anchors = (anchor_result("B0 RandomLegal", prob_lt_guardrail=0.04),)
    overall = posterior(prob_gt_target=0.96)

    assert (
        _decision_reasons(
            anchor_results=anchors,
            overall=overall,
            truncation=PromotionGateRate(numerator=1, denominator=20, rate=0.05),
            stack=stack,
        )
        == []
    )

    above_limit = _decision_reasons(
        anchor_results=anchors,
        overall=overall,
        truncation=PromotionGateRate(numerator=2, denominator=20, rate=0.1),
        stack=stack,
    )
    assert {reason["code"] for reason in above_limit} == {"truncation_rate_exceeded"}
