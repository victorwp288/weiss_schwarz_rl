from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.learning_run_compare import (
    build_run_learning_comparison,
    load_dev_eval_records,
    load_learning_progress_metrics,
    load_policy_alignment_metrics,
)


def _write_dev_eval_log(run_dir: Path, payload: dict[str, object]) -> None:
    path = run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_learning_progress_summary(run_dir: Path, payload: dict[str, object]) -> None:
    path = run_dir / "diagnostics" / "learning_progress_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dev_eval_summary(run_dir: Path, update_count: int, payload: dict[str, object]) -> None:
    path = run_dir / "eval" / "dev_eval" / f"update_{update_count}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learning_run_compare_flags_cross_seed_anchor_fragility(tmp_path: Path) -> None:
    run_a = tmp_path / "seed_a"
    run_b = tmp_path / "seed_b"
    _write_dev_eval_log(
        run_a,
        {
            "train_u25_p1": {
                "update_count": 25,
                "aggregate_score": 0.70,
                "anchor_scores": {"B0 RandomLegal": 1.0, "B2 HeuristicPublic": 0.40},
            },
            "train_u50_p2": {
                "update_count": 50,
                "aggregate_score": 0.82,
                "anchor_scores": {"B0 RandomLegal": 1.0, "B2 HeuristicPublic": 0.64},
            },
        },
    )
    _write_dev_eval_log(
        run_b,
        {
            "train_u25_p1": {
                "update_count": 25,
                "aggregate_score": 0.45,
                "anchor_scores": {"B0 RandomLegal": 0.85, "B2 HeuristicPublic": 0.05},
            },
            "train_u50_p2": {
                "update_count": 50,
                "aggregate_score": 0.42,
                "anchor_scores": {"B0 RandomLegal": 0.84, "B2 HeuristicPublic": 0.0},
            },
        },
    )

    summary = build_run_learning_comparison(
        [run_a, run_b],
        fragility_threshold=0.25,
        anchor_fragility_threshold=0.25,
    )

    assert summary["run_count"] == 2
    assert summary["runs"][0]["best"]["update_count"] == 50
    assert summary["runs"][1]["best"]["update_count"] == 25
    update_50 = summary["by_update"]["50"]
    assert update_50["aggregate"]["range"] == pytest.approx(0.40)
    assert update_50["anchors"]["B2 HeuristicPublic"]["range"] == pytest.approx(0.64)
    assert any("aggregate seed/run fragility at update 50" in warning for warning in summary["warnings"])
    assert any("B2 HeuristicPublic seed/run fragility at update 50" in warning for warning in summary["warnings"])


def test_learning_run_compare_falls_back_to_eval_dev_eval_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    summary_path = run_dir / "eval" / "dev_eval" / "update_25" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "policy_id": "train_u25_p1",
                "update_count": 25,
                "aggregate_score": 0.5,
                "anchor_scores": {"B0 RandomLegal": 1.0, "B2 HeuristicPublic": 0.0},
            }
        ),
        encoding="utf-8",
    )

    records = load_dev_eval_records(run_dir)

    assert records == [
        {
            "policy_id": "train_u25_p1",
            "update_count": 25,
            "aggregate_score": 0.5,
            "anchor_scores": {"B0 RandomLegal": 1.0, "B2 HeuristicPublic": 0.0},
        }
    ]


def test_learning_run_compare_includes_learning_progress_metric_ranges(tmp_path: Path) -> None:
    run_a = tmp_path / "teacherfade"
    run_b = tmp_path / "postbest"
    _write_dev_eval_log(
        run_a,
        {
            "train_u50_p10": {
                "update_count": 50,
                "aggregate_score": 0.71,
                "anchor_scores": {"B2 HeuristicPublic": 0.56},
            }
        },
    )
    _write_dev_eval_log(
        run_b,
        {
            "train_u25_p5": {
                "update_count": 25,
                "aggregate_score": 0.64,
                "anchor_scores": {"B2 HeuristicPublic": 0.53},
            }
        },
    )
    _write_learning_progress_summary(
        run_a,
        {
            "off_policy": {
                "max_vtrace_rho_p99": 4.0,
                "max_target_behavior_train_logp_delta_abs_p99": 0.9,
            },
            "chosen_action_learning": {
                "chosen_pass_train_advantage_mean": {"last_window_mean": -0.03},
                "chosen_nonpass_train_advantage_mean": {"last_window_mean": 0.02},
            },
            "teacher_action_accuracy": {"last_window_mean": 0.72},
            "teacher_guidance": {"max_teacher_public_heuristic_coef_active": 0.04},
        },
    )
    _write_dev_eval_summary(
        run_a,
        50,
        {
            "anchors": {
                "B4 HeuristicPublicControl": {
                    "policy_alignment_diagnostics": {
                        "focal_policy_turns": {
                            "model_matches_reference_top_action_rate": 0.70,
                            "model_matches_reference_top_action_family_rate": 0.99,
                            "model_mean_probability_on_reference_top_action": 0.55,
                            "model_mean_probability_on_reference_top_action_family": 0.90,
                            "reference_top_family_summaries": [
                                {
                                    "family": "main_play_character",
                                    "model_matches_reference_top_action_rate": 0.40,
                                    "model_mean_probability_on_reference_top_action": 0.20,
                                    "model_reference_top_action_same_family_logit_margin_percentiles": {"mean": 0.10},
                                }
                            ],
                        }
                    }
                }
            }
        },
    )
    _write_learning_progress_summary(
        run_b,
        {
            "off_policy": {
                "max_vtrace_rho_p99": 9.5,
                "max_target_behavior_train_logp_delta_abs_p99": 1.4,
            },
            "chosen_action_learning": {
                "chosen_pass_train_advantage_mean": {"last_window_mean": -0.05},
                "chosen_nonpass_train_advantage_mean": {"last_window_mean": 0.01},
            },
            "teacher_action_accuracy": {"last_window_mean": 0.69},
            "teacher_guidance": {"max_teacher_public_heuristic_coef_active": 0.04},
        },
    )
    _write_dev_eval_summary(
        run_b,
        25,
        {
            "anchors": {
                "B4 HeuristicPublicControl": {
                    "policy_alignment_diagnostics": {
                        "focal_policy_turns": {
                            "model_matches_reference_top_action_rate": 0.62,
                            "model_matches_reference_top_action_family_rate": 0.98,
                            "model_mean_probability_on_reference_top_action": 0.50,
                            "model_mean_probability_on_reference_top_action_family": 0.88,
                            "reference_top_family_summaries": [
                                {
                                    "family": "main_play_character",
                                    "model_matches_reference_top_action_rate": 0.28,
                                    "model_mean_probability_on_reference_top_action": 0.15,
                                    "model_reference_top_action_same_family_logit_margin_percentiles": {"mean": -0.20},
                                }
                            ],
                        }
                    }
                }
            }
        },
    )

    assert load_learning_progress_metrics(run_a)["off_policy_max_vtrace_rho_p99"] == pytest.approx(4.0)
    assert load_policy_alignment_metrics(run_a, update_count=50)[
        "b4_focal_main_play_character_top_action_rate"
    ] == pytest.approx(0.40)

    summary = build_run_learning_comparison([run_a, run_b])

    first_run_metrics = summary["runs"][0]["diagnostic_metrics"]
    assert first_run_metrics["chosen_pass_advantage_last_window"] == pytest.approx(-0.03)
    rho_range = summary["diagnostic_metric_ranges"]["off_policy_max_vtrace_rho_p99"]
    assert rho_range["min_run"] == "teacherfade"
    assert rho_range["max_run"] == "postbest"
    assert rho_range["range"] == pytest.approx(5.5)
    b4_main_play_range = summary["policy_alignment_metric_ranges"]["b4_focal_main_play_character_top_action_rate"]
    assert b4_main_play_range["min_run"] == "postbest"
    assert b4_main_play_range["max_run"] == "teacherfade"
    assert b4_main_play_range["range"] == pytest.approx(0.12)
