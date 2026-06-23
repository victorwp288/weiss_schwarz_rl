from __future__ import annotations

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.training.train_entrypoint import (
    _checkpoint_candidate_metric,
    _should_promote_best_checkpoint,
)

from tests.weiss_rl._config_paths import repo_root


def test_checkpoint_candidate_metric_prefers_aggregate_score() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 1.0},
        dev_eval_summary={
            "aggregate_score": 0.625,
            "uncertainty": {"mean": 0.125},
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert metric_kind == "dev_eval_mean"
    assert metric_value == pytest.approx(0.625)


def test_checkpoint_candidate_metric_rejects_truncation_heavy_dev_eval() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.75},
        dev_eval_summary={
            "aggregate_score": 0.8,
            "stall_monitor": {"worst_truncation_rate": 0.5},
        },
    )

    assert metric_kind is None
    assert metric_value is None


def test_checkpoint_candidate_metric_rejects_low_confidence_dev_eval() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={
            "aggregate_score": 0.8,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 0.95, "prob_lt_half": 0.05, "ci_half_width": 0.1}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.52, "prob_lt_half": 0.48, "ci_half_width": 0.28}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert metric_kind is None
    assert metric_value is None


def test_checkpoint_candidate_metric_waits_for_dev_eval_when_periodic_eval_enabled() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.5},
        dev_eval_summary=None,
    )

    assert metric_kind is None
    assert metric_value is None


def test_should_not_promote_best_checkpoint_when_candidate_metric_is_missing() -> None:
    assert (
        _should_promote_best_checkpoint(
            existing_record=None,
            candidate_kind=None,
            candidate_value=None,
        )
        is False
    )
