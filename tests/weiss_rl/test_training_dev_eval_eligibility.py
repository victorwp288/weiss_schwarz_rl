from __future__ import annotations

import pytest
import weiss_rl.training.checkpointing.guard as checkpoint_guard
from weiss_rl.config import apply_stack_overrides, load_stack_config
from weiss_rl.training.train_entrypoint import (
    _dev_eval_ineligibility_reasons,
)

from tests.weiss_rl._config_paths import repo_root


def test_dev_eval_ineligibility_reasons_identify_borderline_confidence_only() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.625,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.1467}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.686, "prob_lt_half": 0.314, "ci_half_width": 0.2492}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert reasons == ("confidence_ci",)


def test_dev_eval_assessment_collects_timeout_rates_and_confidence() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    assessment = checkpoint_guard.assess_dev_eval_metric_eligibility(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.625,
            "anchors": {
                "B0 RandomLegal": {
                    "summary": {"games": 10, "truncations": 1, "no_progress_timeouts": 2, "natural_timeouts": 3},
                    "uncertainty": {"prob_gt_half": 0.9, "prob_lt_half": 0.1, "ci_half_width": 0.08},
                },
                "B1 NoLeague baseline": {
                    "summary": {"games": 20, "truncations": 5, "no_progress_timeouts": 1, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.55, "prob_lt_half": 0.45, "ci_half_width": 0.30},
                },
            },
        },
    )

    assert assessment.score == pytest.approx(0.625)
    assert assessment.timeout_rates.worst_truncation_rate == pytest.approx(0.25)
    assert assessment.timeout_rates.worst_no_progress_timeout_rate == pytest.approx(0.20)
    assert assessment.timeout_rates.worst_natural_timeout_rate == pytest.approx(0.30)
    assert assessment.timeout_rates.worst_stall_rate == pytest.approx(0.20)
    assert assessment.confidence == checkpoint_guard.DevEvalConfidenceStats(
        min_prob_gt_half=0.55,
        max_prob_lt_half=0.45,
        max_ci_half_width=0.30,
    )
    assert assessment.reasons == ("confidence_prob", "confidence_ci")
    assert not assessment.eligible
    assert checkpoint_guard.dev_eval_confidence_stats(
        {
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 0.9, "prob_lt_half": 0.1, "ci_half_width": 0.08}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.55, "prob_lt_half": 0.45, "ci_half_width": 0.30}
                },
            }
        }
    ) == {
        "min_prob_gt_half": 0.55,
        "max_prob_lt_half": 0.45,
        "max_ci_half_width": 0.30,
    }


def test_dev_eval_ineligibility_reasons_apply_checkpoint_confidence_when_stall_monitor_disabled() -> None:
    stack = apply_stack_overrides(
        load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml"),
        {"curriculum.stall_monitor.enabled": False},
    )
    assert stack.config.curriculum is not None
    assert stack.config.curriculum.stall_monitor.enabled is False
    assert stack.config.curriculum.checkpoint_guard.enabled is True

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.640625,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.07}},
                "B2 HeuristicPublic": {
                    "uncertainty": {"prob_gt_half": 0.021, "prob_lt_half": 0.979, "ci_half_width": 0.16}
                },
            },
        },
    )

    assert reasons == ("confidence_prob",)
