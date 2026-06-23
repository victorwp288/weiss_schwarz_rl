from __future__ import annotations

import pytest
from weiss_rl.config import apply_stack_overrides, load_stack_config
from weiss_rl.training.train_entrypoint import (
    _confirmatory_dev_eval_request,
    _expand_periodic_dev_eval_paired_seeds,
)

from tests.weiss_rl._config_paths import repo_root


def test_confirmatory_dev_eval_request_targets_borderline_score_drop_for_reevaluation() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.84375},
        dev_eval_summary={
            "aggregate_score": 0.71875,
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is not None
    assert request["reasons"] == ["score_drop"]
    assert request["current_score"] == pytest.approx(0.71875)
    assert request["existing_best_score"] == pytest.approx(0.84375)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_targets_score_improving_borderline_candidate() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.59375},
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

    assert request is not None
    assert request["reasons"] == ["confidence_ci"]
    assert request["current_score"] == pytest.approx(0.625)
    assert request["existing_best_score"] == pytest.approx(0.59375)
    assert request["ci_excess"] == pytest.approx(0.0092, abs=1e-4)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_targets_multianchor_near_miss_candidate() -> None:
    stack = apply_stack_overrides(
        load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml"),
        {
            "league.promotion.anchor_set_v1.required": [
                "B0 RandomLegal",
                "B2 HeuristicPublic",
                "B3 HeuristicPublicAggro",
                "B4 HeuristicPublicControl",
            ]
        },
    )

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=None,
        dev_eval_summary={
            "aggregate_score": 0.6953125,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"mean": 1.0, "prob_gt_half": 1.0, "ci_half_width": 0.0}},
                "B2 HeuristicPublic": {"uncertainty": {"mean": 0.6875, "prob_gt_half": 0.989, "ci_half_width": 0.14}},
                "B3 HeuristicPublicAggro": {
                    "uncertainty": {"mean": 0.625, "prob_gt_half": 0.899, "ci_half_width": 0.17}
                },
                "B4 HeuristicPublicControl": {
                    "uncertainty": {"mean": 0.46875, "prob_gt_half": 0.359, "ci_half_width": 0.18}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is not None
    assert request["reasons"] == ["confidence_prob"]
    assert request["current_score"] == pytest.approx(0.6953125)
    assert request["worst_anchor_mean"] == pytest.approx(0.46875)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_rejects_multianchor_clear_anchor_failure() -> None:
    stack = apply_stack_overrides(
        load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml"),
        {
            "league.promotion.anchor_set_v1.required": [
                "B0 RandomLegal",
                "B2 HeuristicPublic",
                "B3 HeuristicPublicAggro",
                "B4 HeuristicPublicControl",
            ]
        },
    )

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=None,
        dev_eval_summary={
            "aggregate_score": 0.6171875,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"mean": 1.0, "prob_gt_half": 1.0, "ci_half_width": 0.0}},
                "B2 HeuristicPublic": {"uncertainty": {"mean": 0.65625, "prob_gt_half": 0.962, "ci_half_width": 0.16}},
                "B3 HeuristicPublicAggro": {
                    "uncertainty": {"mean": 0.4375, "prob_gt_half": 0.212, "ci_half_width": 0.17}
                },
                "B4 HeuristicPublicControl": {
                    "uncertainty": {"mean": 0.375, "prob_gt_half": 0.03, "ci_half_width": 0.13}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is None


def test_expand_periodic_dev_eval_paired_seeds_is_deterministic_and_unique() -> None:
    base_paired_seeds = list(range(8))

    expanded_a = _expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=32,
        seed_file_sha256="abc123",
        update_count=200,
        policy_version=10,
        scope="periodic_dev_eval_confirmatory",
    )
    expanded_b = _expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=32,
        seed_file_sha256="abc123",
        update_count=200,
        policy_version=10,
        scope="periodic_dev_eval_confirmatory",
    )

    assert expanded_a[:8] == base_paired_seeds
    assert expanded_a == expanded_b
    assert len(expanded_a) == 32
    assert len(set(expanded_a)) == 32
