from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.diagnostics import learning_progress as learning_progress_module
from weiss_rl.diagnostics.learning_progress import build_learning_progress_summary, evaluate_league_guard
from weiss_rl.diagnostics.learning_progress_artifacts import (
    _final_eval_matrix_summary as artifact_final_eval_matrix_summary,
)
from weiss_rl.diagnostics.learning_progress_artifacts import (
    _periodic_dev_eval_trend as artifact_periodic_dev_eval_trend,
)
from weiss_rl.diagnostics.learning_progress_artifacts import _promotion_gate_summary as artifact_promotion_gate_summary
from weiss_rl.diagnostics.learning_progress_guard import (
    DEFAULT_LEAGUE_GUARD_ANCHORS as guard_default_league_guard_anchors,
)
from weiss_rl.diagnostics.learning_progress_guard import (
    evaluate_league_guard as guard_evaluate_league_guard,
)
from weiss_rl.diagnostics.learning_progress_metrics import (
    _window_summary as metric_window_summary,
)
from weiss_rl.diagnostics.learning_progress_metrics import (
    build_training_log_summary_sections as metric_build_training_log_summary_sections,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learning_progress_artifact_helpers_are_package_owned() -> None:
    assert learning_progress_module._final_eval_matrix_summary is artifact_final_eval_matrix_summary
    assert learning_progress_module._periodic_dev_eval_trend is artifact_periodic_dev_eval_trend
    assert learning_progress_module._promotion_gate_summary is artifact_promotion_gate_summary
    assert artifact_final_eval_matrix_summary.__module__ == "weiss_rl.diagnostics.learning_progress_artifacts"


def test_learning_progress_guard_helpers_are_package_owned() -> None:
    assert evaluate_league_guard is guard_evaluate_league_guard
    assert learning_progress_module.DEFAULT_LEAGUE_GUARD_ANCHORS is guard_default_league_guard_anchors
    assert guard_evaluate_league_guard.__module__ == "weiss_rl.diagnostics.learning_progress_guard"


def test_learning_progress_metric_sections_are_package_owned() -> None:
    assert learning_progress_module._window_summary is metric_window_summary
    assert learning_progress_module.build_training_log_summary_sections is metric_build_training_log_summary_sections
    assert metric_build_training_log_summary_sections.__module__ == "weiss_rl.diagnostics.learning_progress_metrics"


def test_learning_progress_diagnostic_flags_large_vtrace_rho(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "training" / "logs" / "training_metrics.jsonl",
        [
            {
                "update_count": 1,
                "loss": 1.0,
                "vtrace_rho_mean": 1.0,
                "vtrace_rho_p99": 2.5,
                "vtrace_clip_rate": 0.1,
                "custom_metrics": {
                    "reward_abs_mean": 0.1,
                    "reward_mean": 0.02,
                    "reward_nonzero_fraction": 0.5,
                    "reward_positive_fraction": 0.3,
                    "reward_std": 0.2,
                    "target_abs_mean": 0.4,
                    "chosen_mulligan_confirm_train_fraction": 0.02,
                    "chosen_mulligan_select_train_fraction": 0.18,
                    "chosen_mulligan_confirm_train_advantage_mean": 0.1,
                    "chosen_mulligan_select_train_advantage_mean": -0.3,
                    "teacher_aux_loss": 0.12,
                    "teacher_hand_coef_active": 0.1,
                    "teacher_main_play_character_slot_accuracy": 0.35,
                    "teacher_hand_accuracy": 0.4,
                    "teacher_main_play_character_hand_accuracy": 0.3,
                    "teacher_clock_from_hand_accuracy": 0.2,
                    "teacher_hand_loss": 0.9,
                    "teacher_hand_supported_fraction": 1.0,
                    "teacher_same_family_action_accuracy": 0.2,
                    "teacher_same_family_main_play_character_accuracy": 0.1,
                    "teacher_action_margin_mean": 0.05,
                    "teacher_action_margin_satisfied_fraction": 0.1,
                    "teacher_same_family_action_margin_mean": 0.02,
                    "teacher_same_family_action_margin_satisfied_fraction": 0.05,
                    "teacher_public_heuristic_loss": 2.5,
                    "teacher_public_heuristic_supported_fraction": 0.6,
                    "teacher_public_heuristic_top1_mass": 0.25,
                    "teacher_public_heuristic_target_entropy": 1.7,
                    "policy_anchor_coef_active": 0.08,
                    "policy_anchor_loss": 0.0,
                    "policy_anchor_weighted_loss": 0.0,
                    "policy_anchor_kl_mean": 0.0,
                    "policy_anchor_kl_p95": 0.0,
                    "policy_anchor_top_action_coef_active": 0.04,
                    "policy_anchor_top_action_loss": 0.3,
                    "policy_anchor_top_action_loss_p95": 0.5,
                    "policy_anchor_top_action_agreement": 0.8,
                    "vtrace_train_rho_mean": 2.0,
                    "vtrace_train_rho_p95": 3.0,
                    "vtrace_train_rho_p99": 4.0,
                },
            },
            {
                "update_count": 2,
                "loss": 0.5,
                "vtrace_rho_mean": 2636.0,
                "vtrace_rho_p99": 4100.0,
                "vtrace_clip_rate": 0.75,
                "custom_metrics": {
                    "reward_abs_mean": 0.25,
                    "reward_mean": -0.01,
                    "reward_negative_fraction": 0.4,
                    "reward_nonzero_fraction": 0.9,
                    "reward_std": 0.35,
                    "target_abs_mean": 0.8,
                    "chosen_pass_train_fraction": 0.75,
                    "chosen_pass_train_advantage_mean": -0.2,
                    "chosen_nonpass_train_advantage_mean": 0.4,
                    "chosen_mulligan_confirm_train_fraction": 0.01,
                    "chosen_mulligan_select_train_fraction": 0.09,
                    "chosen_mulligan_confirm_train_advantage_mean": 0.2,
                    "chosen_mulligan_select_train_advantage_mean": -0.4,
                    "teacher_aux_loss": 0.08,
                    "teacher_hand_coef_active": 0.08,
                    "teacher_main_play_character_slot_accuracy": 0.55,
                    "teacher_hand_accuracy": 0.6,
                    "teacher_main_play_character_hand_accuracy": 0.5,
                    "teacher_clock_from_hand_accuracy": 0.45,
                    "teacher_hand_loss": 0.7,
                    "teacher_hand_supported_fraction": 0.9,
                    "teacher_same_family_action_accuracy": 0.3,
                    "teacher_same_family_main_play_character_accuracy": 0.2,
                    "teacher_action_margin_mean": 0.15,
                    "teacher_action_margin_satisfied_fraction": 0.25,
                    "teacher_same_family_action_margin_mean": 0.12,
                    "teacher_same_family_action_margin_satisfied_fraction": 0.2,
                    "teacher_public_heuristic_loss": 2.0,
                    "teacher_public_heuristic_supported_fraction": 0.7,
                    "teacher_public_heuristic_top1_mass": 0.35,
                    "teacher_public_heuristic_target_entropy": 1.4,
                    "policy_anchor_coef_active": 0.08,
                    "policy_anchor_loss": 0.12,
                    "policy_anchor_weighted_loss": 0.0096,
                    "policy_anchor_kl_mean": 0.12,
                    "policy_anchor_kl_p95": 0.2,
                    "policy_anchor_top_action_coef_active": 0.04,
                    "policy_anchor_top_action_loss": 0.25,
                    "policy_anchor_top_action_loss_p95": 0.4,
                    "policy_anchor_top_action_agreement": 0.85,
                    "target_behavior_train_logp_delta_abs_mean": 0.4,
                    "target_behavior_train_logp_delta_abs_p99": 1.5,
                    "vtrace_train_rho_mean": 4108.0,
                    "vtrace_train_rho_p95": 4096.0,
                    "vtrace_train_rho_p99": 4097.0,
                },
            },
        ],
    )
    _write_jsonl(
        run_dir / "training" / "logs" / "scalars.jsonl",
        [
            {
                "update_count": 1,
                "teacher_public_heuristic_coef_active": 0.05,
                "collector_teacher_tactical_row_count": 10,
                "collector_total_actions": 100,
            },
            {
                "update_count": 2,
                "teacher_public_heuristic_coef_active": 0.04,
                "collector_teacher_tactical_row_count": 30,
                "collector_total_actions": 200,
            },
        ],
    )

    summary = build_learning_progress_summary(run_dir)

    assert summary["off_policy"]["max_vtrace_rho_mean"] == 2636.0
    assert summary["off_policy"]["max_vtrace_rho_p99"] == 4100.0
    assert summary["off_policy"]["max_vtrace_train_rho_mean"] == 4108.0
    assert summary["off_policy"]["max_vtrace_train_rho_p95"] == 4096.0
    assert summary["off_policy"]["max_vtrace_train_rho_p99"] == 4097.0
    assert summary["off_policy"]["max_vtrace_clip_rate"] == 0.75
    assert summary["reward_scale"]["reward_abs_mean"]["last"] == 0.25
    assert summary["reward_scale"]["reward_positive_fraction"]["last"] == 0.3
    assert summary["reward_scale"]["reward_negative_fraction"]["last"] == 0.4
    assert summary["reward_scale"]["max_reward_abs_mean"] == 0.25
    assert summary["reward_scale"]["max_target_abs_mean"] == 0.8
    assert summary["chosen_action_learning"]["chosen_pass_train_fraction"]["last"] == pytest.approx(0.75)
    assert summary["chosen_action_learning"]["chosen_pass_train_advantage_mean"]["last"] == pytest.approx(-0.2)
    assert summary["chosen_action_learning"]["chosen_nonpass_train_advantage_mean"]["last"] == pytest.approx(0.4)
    assert summary["chosen_action_learning"]["chosen_mulligan_confirm_train_fraction"]["last"] == pytest.approx(0.01)
    assert summary["chosen_action_learning"]["chosen_mulligan_select_train_fraction"]["last"] == pytest.approx(0.09)
    assert summary["chosen_action_learning"]["chosen_mulligan_select_share_of_mulligan"]["last"] == pytest.approx(0.9)
    assert summary["chosen_action_learning"]["chosen_mulligan_confirm_train_advantage_mean"]["last"] == pytest.approx(
        0.2
    )
    assert summary["chosen_action_learning"]["chosen_mulligan_select_train_advantage_mean"]["last"] == pytest.approx(
        -0.4
    )
    teacher = summary["teacher_guidance"]
    assert teacher["teacher_public_heuristic_coef_active"]["last"] == pytest.approx(0.04)
    assert teacher["teacher_hand_coef_active"]["last"] == pytest.approx(0.08)
    assert teacher["teacher_aux_loss"]["last"] == pytest.approx(0.08)
    assert teacher["teacher_main_play_character_slot_accuracy"]["last"] == pytest.approx(0.55)
    assert teacher["teacher_hand_accuracy"]["last"] == pytest.approx(0.6)
    assert teacher["teacher_main_play_character_hand_accuracy"]["last"] == pytest.approx(0.5)
    assert teacher["teacher_clock_from_hand_accuracy"]["last"] == pytest.approx(0.45)
    assert teacher["teacher_hand_loss"]["last"] == pytest.approx(0.7)
    assert teacher["teacher_hand_supported_fraction"]["last"] == pytest.approx(0.9)
    assert teacher["teacher_same_family_action_accuracy"]["last"] == pytest.approx(0.3)
    assert teacher["teacher_same_family_main_play_character_accuracy"]["last"] == pytest.approx(0.2)
    assert teacher["teacher_action_margin_mean"]["last"] == pytest.approx(0.15)
    assert teacher["teacher_action_margin_satisfied_fraction"]["last"] == pytest.approx(0.25)
    assert teacher["teacher_same_family_action_margin_mean"]["last"] == pytest.approx(0.12)
    assert teacher["teacher_same_family_action_margin_satisfied_fraction"]["last"] == pytest.approx(0.2)
    assert teacher["teacher_public_heuristic_loss"]["last"] == pytest.approx(2.0)
    assert teacher["teacher_public_heuristic_supported_fraction"]["last"] == pytest.approx(0.7)
    assert teacher["teacher_public_heuristic_top1_mass"]["last"] == pytest.approx(0.35)
    assert teacher["teacher_public_heuristic_target_entropy"]["last"] == pytest.approx(1.4)
    assert teacher["teacher_tactical_row_fraction_of_total"]["last"] == pytest.approx(0.15)
    assert teacher["policy_anchor_coef_active"]["last"] == pytest.approx(0.08)
    assert teacher["policy_anchor_loss"]["last"] == pytest.approx(0.12)
    assert teacher["policy_anchor_weighted_loss"]["last"] == pytest.approx(0.0096)
    assert teacher["policy_anchor_kl_mean"]["last"] == pytest.approx(0.12)
    assert teacher["policy_anchor_kl_p95"]["last"] == pytest.approx(0.2)
    assert teacher["policy_anchor_top_action_coef_active"]["last"] == pytest.approx(0.04)
    assert teacher["policy_anchor_top_action_loss"]["last"] == pytest.approx(0.25)
    assert teacher["policy_anchor_top_action_loss_p95"]["last"] == pytest.approx(0.4)
    assert teacher["policy_anchor_top_action_agreement"]["last"] == pytest.approx(0.85)
    assert summary["off_policy"]["max_target_behavior_train_logp_delta_abs_mean"] == pytest.approx(0.4)
    assert summary["off_policy"]["max_target_behavior_train_logp_delta_abs_p99"] == pytest.approx(1.5)
    assert any("vtrace_rho_mean exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_rho_p99 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_mean exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_p95 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_train_rho_p99 exceeded 10" in warning for warning in summary["warnings"])
    assert any("vtrace_clip_rate exceeded 0.5" in warning for warning in summary["warnings"])
    assert any("mulligan-confirm collapse suspected" in warning for warning in summary["warnings"])
    assert any("target_behavior_train_logp_delta_abs_p99 exceeded" in warning for warning in summary["warnings"])


def test_learning_progress_diagnostic_summarizes_periodic_dev_eval_trend(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "training" / "logs" / "training_metrics.jsonl",
        [
            {"update_count": 25, "loss": 1.0, "vtrace_rho_p99": 1.0},
            {"update_count": 50, "loss": 0.5, "vtrace_rho_p99": 4.0},
            {"update_count": 75, "loss": 0.4, "vtrace_rho_p99": 2.0},
        ],
    )
    _write_jsonl(
        run_dir / "training" / "logs" / "scalars.jsonl",
        [
            {
                "update_count": 25,
                "league_update_lag": 24.0,
                "policy_version_lag_p50": 0.0,
                "policy_version_lag_p90": 0.0,
                "learner_actor_update_lag_p50": 24.0,
                "learner_actor_update_lag_p90": 24.0,
            },
            {
                "update_count": 50,
                "league_update_lag": 49.0,
                "policy_version_lag_p50": 0.0,
                "policy_version_lag_p90": 0.0,
                "learner_actor_update_lag_p50": 49.0,
                "learner_actor_update_lag_p90": 49.0,
            },
            {
                "update_count": 75,
                "league_update_lag": 24.0,
                "policy_version_lag_p50": 0.0,
                "policy_version_lag_p90": 0.0,
                "learner_actor_update_lag_p50": 24.0,
                "learner_actor_update_lag_p90": 24.0,
            },
        ],
    )
    dev_eval_path = run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json"
    dev_eval_path.parent.mkdir(parents=True, exist_ok=True)
    dev_eval_path.write_text(
        json.dumps(
            {
                "train_u25_p1": {"update_count": 25, "aggregate_score": 0.45, "anchor_scores": {"B0": 0.8}},
                "train_u50_p2": {"update_count": 50, "aggregate_score": 0.60, "anchor_scores": {"B0": 0.9}},
                "train_u75_p3": {"update_count": 75, "aggregate_score": 0.52, "anchor_scores": {"B0": 0.85}},
            }
        ),
        encoding="utf-8",
    )

    summary = build_learning_progress_summary(run_dir)

    trend = summary["periodic_dev_eval"]
    assert trend["best_update"] == 50
    assert trend["best_aggregate_score"] == 0.60
    assert trend["last_update"] == 75
    assert trend["last_aggregate_score"] == 0.52
    assert trend["latest_minus_best"] == pytest.approx(-0.08)
    assert trend["non_monotonic_drop_count"] == 1
    assert summary["actor_model_sync"]["max_policy_version_lag_p90"] == 0.0
    assert summary["actor_model_sync"]["max_learner_actor_update_lag_p90"] == 49.0
    assert summary["actor_model_sync"]["max_learner_to_actor_update_lag"] == 49.0
    assert summary["league_sync"]["max_league_update_lag"] == 49.0
    assert summary["off_policy"]["stale_policy_lag_source"] == "learner_actor_update_lag_p90"
    assert summary["off_policy"]["stale_policy_lag_correlations"]["vtrace_rho_p99"]["paired_update_count"] == 3
    assert summary["off_policy"]["stale_policy_lag_correlations"]["vtrace_rho_p99"]["pearson"] == pytest.approx(
        0.94491118
    )
    assert any("latest periodic dev-eval aggregate" in warning for warning in summary["warnings"])
    assert any("learner_actor_update_lag_p90 exceeded" in warning for warning in summary["warnings"])


def test_learning_progress_diagnostic_summarizes_promotion_gate_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "training" / "logs" / "training_metrics.jsonl", [{"update_count": 10, "loss": 1.0}])
    _write_jsonl(
        run_dir / "training" / "logs" / "scalars.jsonl",
        [
            {
                "update_count": 20,
                "pfsp_pool_size": 4,
                "pfsp_champion_pool_size": 0,
                "pfsp_recent_pool_size": 4,
                "pfsp_hard_negative_pool_size": 0,
                "pfsp_quarantined_opponents": 0,
                "pfsp_sampled_envs": 720,
                "pfsp_mirror_envs": 480,
                "pfsp_champion_envs": 0,
                "pfsp_recent_envs": 480,
                "pfsp_hard_negative_envs": 0,
                "pfsp_warmup_snapshot_envs": 0,
            }
        ],
    )
    _write_json(
        run_dir / "eval" / "promotion_gate" / "update_10" / "promotion_gate.json",
        {
            "focal_policy_id": "policy_000002",
            "decision": {
                "passed": False,
                "reasons": [
                    {"code": "overall_posterior_below_threshold"},
                    {"code": "anchor_loss_guardrail_exceeded", "anchor_name": "B4 HeuristicPublicControl"},
                ],
            },
            "overall_posterior": {"mean": 0.5, "prob_gt_target": 0.1},
            "anchors": [
                {"anchor_name": "B2 HeuristicPublic", "posterior": {"mean": 0.5}},
                {"anchor_name": "B4 HeuristicPublicControl", "posterior": {"mean": 0.375}},
            ],
        },
    )
    _write_json(
        run_dir / "eval" / "promotion_gate" / "update_15" / "promotion_gate.json",
        {
            "focal_policy_id": "policy_000003",
            "decision": {
                "passed": False,
                "reasons": [{"code": "anchor_loss_guardrail_exceeded", "anchor_name": "B3 HeuristicPublicAggro"}],
            },
            "overall_posterior": {"mean": 0.5625, "prob_gt_target": 0.4},
            "anchors": [{"anchor_name": "B3 HeuristicPublicAggro", "posterior": {"mean": 0.375}}],
        },
    )
    _write_json(
        run_dir / "eval" / "promotion_gate" / "update_20" / "promotion_gate.json",
        {
            "focal_policy_id": "policy_000004",
            "decision": {
                "passed": False,
                "reasons": [{"code": "anchor_loss_guardrail_exceeded", "anchor_name": "B4 HeuristicPublicControl"}],
            },
            "overall_posterior": {"mean": 0.5625, "prob_gt_target": 0.5},
            "anchors": [{"anchor_name": "B4 HeuristicPublicControl", "posterior": {"mean": 0.375}}],
        },
    )

    summary = build_learning_progress_summary(run_dir)
    gate = summary["promotion_gate"]

    assert gate["attempt_count"] == 3
    assert gate["passed_count"] == 0
    assert gate["failed_count"] == 3
    assert gate["first_pass_update"] is None
    assert gate["latest_update"] == 20
    assert gate["latest_passed"] is False
    assert gate["latest_reason_codes"] == ["anchor_loss_guardrail_exceeded"]
    assert gate["consecutive_failure_count"] == 3
    assert gate["records"][0]["anchor_means"]["B4 HeuristicPublicControl"] == pytest.approx(0.375)
    assert summary["league_sampling"]["latest_has_admitted_champion"] is False
    assert summary["league_sampling"]["latest_probationary_recent_sampling_active"] is True
    assert summary["league_sampling"]["snapshot_env_fraction"]["last"] == pytest.approx(0.4)
    assert any("promotion gate never passed" in warning for warning in summary["warnings"])
    assert any("probationary snapshot sampling was active" in warning for warning in summary["warnings"])
    assert any("promotion gate failed 3 consecutive attempts" in warning for warning in summary["warnings"])


def test_learning_progress_diagnostic_league_guard_fails_on_anchor_and_promotion_health(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "training" / "logs" / "training_metrics.jsonl",
        [{"update_count": 20, "loss": 1.0, "vtrace_rho_p99": 31.0}],
    )
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u10_p2": {
                "update_count": 10,
                "aggregate_score": 0.62,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.55,
                    "B3 HeuristicPublicAggro": 0.52,
                    "B4 HeuristicPublicControl": 0.50,
                },
            },
            "train_u20_p4": {
                "update_count": 20,
                "aggregate_score": 0.50,
                "anchor_scores": {
                    "B2 HeuristicPublic": 0.34,
                    "B3 HeuristicPublicAggro": 0.41,
                    "B4 HeuristicPublicControl": 0.47,
                },
            },
        },
    )
    for update in (10, 15, 20):
        _write_json(
            run_dir / "eval" / "promotion_gate" / f"update_{update}" / "promotion_gate.json",
            {
                "focal_policy_id": f"policy_{update:06d}",
                "decision": {"passed": False, "reasons": [{"code": "anchor_loss_guardrail_exceeded"}]},
                "overall_posterior": {"mean": 0.5, "prob_gt_target": 0.1},
            },
        )

    summary = build_learning_progress_summary(run_dir)
    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is False
    failure_codes = {failure["code"] for failure in guard["failures"]}
    assert "latest_anchor_below_threshold" in failure_codes
    assert "latest_periodic_drop_exceeded" in failure_codes
    assert "promotion_gate_no_pass_after_attempts" in failure_codes
    assert "promotion_gate_consecutive_failures_exceeded" in failure_codes
    assert "vtrace_rho_p99_exceeded" in failure_codes
    b2_failure = next(
        failure
        for failure in guard["failures"]
        if failure["code"] == "latest_anchor_below_threshold" and failure["anchor"] == "B2 HeuristicPublic"
    )
    assert b2_failure["observed"] == pytest.approx(0.34)
    assert guard["latest_anchor_scores"]["B4 HeuristicPublicControl"] == pytest.approx(0.47)


def test_learning_progress_diagnostic_league_guard_passes_on_healthy_probe() -> None:
    summary = {
        "periodic_dev_eval": {
            "latest_minus_best": -0.01,
            "records": [
                {
                    "update_count": 20,
                    "aggregate_score": 0.62,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.55,
                        "B3 HeuristicPublicAggro": 0.53,
                        "B4 HeuristicPublicControl": 0.51,
                    },
                }
            ],
        },
        "promotion_gate": {
            "attempt_count": 3,
            "passed_count": 1,
            "consecutive_failure_count": 1,
        },
        "off_policy": {"max_vtrace_rho_p99": 12.0},
    }

    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is True
    assert guard["failures"] == []


def test_learning_progress_diagnostic_league_guard_uses_train_vtrace_tail_when_available() -> None:
    summary = {
        "periodic_dev_eval": {
            "latest_minus_best": -0.01,
            "records": [
                {
                    "update_count": 20,
                    "aggregate_score": 0.62,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.55,
                        "B3 HeuristicPublicAggro": 0.53,
                        "B4 HeuristicPublicControl": 0.51,
                    },
                }
            ],
        },
        "promotion_gate": {
            "attempt_count": 3,
            "passed_count": 1,
            "consecutive_failure_count": 1,
        },
        "off_policy": {
            "max_vtrace_rho_p99": 31.0,
            "max_vtrace_train_rho_p99": 2.0,
        },
    }

    guard = evaluate_league_guard(summary, max_vtrace_rho_p99=25.0)

    assert guard["passed"] is True
    assert guard["failures"] == []
    assert guard["vtrace_guard_tail_source"] == "train"


def test_learning_progress_diagnostic_summarizes_action_distribution(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "training" / "logs" / "training_metrics.jsonl",
        [{"update_count": 1, "loss": 1.0}, {"update_count": 2, "loss": 0.8}],
    )
    _write_jsonl(
        run_dir / "training" / "logs" / "scalars.jsonl",
        [
            {
                "update_count": 1,
                "collector_total_actions": 100,
                "collector_main_move_actions": 7,
                "collector_pass_actions": 30,
                "collector_pass_with_nonpass_available": 12,
                "collector_mulligan_select_with_confirm_penalty_count": 4,
                "collector_main_move_only_force_pass_rows": 2,
                "collector_main_move_only_force_pass_actions": 5,
                "collector_max_consecutive_main_moves": 1,
            },
            {
                "update_count": 2,
                "collector_total_actions": 200,
                "collector_main_move_actions": 20,
                "collector_pass_actions": 80,
                "collector_pass_with_nonpass_available": 40,
                "collector_mulligan_select_with_confirm_penalty_count": 10,
                "collector_main_move_only_force_pass_rows": 6,
                "collector_main_move_only_force_pass_actions": 14,
                "collector_max_consecutive_main_moves": 2,
            },
        ],
    )

    summary = build_learning_progress_summary(run_dir)

    actions = summary["action_distribution"]
    assert actions["main_move_fraction"]["first"] == pytest.approx(0.07)
    assert actions["main_move_fraction"]["last"] == pytest.approx(0.10)
    assert actions["pass_fraction"]["last"] == pytest.approx(0.40)
    assert actions["pass_with_nonpass_fraction_of_total"]["last"] == pytest.approx(0.20)
    assert actions["pass_with_nonpass_fraction_of_pass"]["last"] == pytest.approx(0.50)
    assert actions["mulligan_select_with_confirm_penalty_fraction_of_total"]["last"] == pytest.approx(0.05)
    assert actions["main_move_only_force_pass_rows_fraction_of_total"]["last"] == pytest.approx(0.03)
    assert actions["main_move_only_force_pass_actions_fraction_of_total"]["last"] == pytest.approx(0.07)
    assert actions["max_consecutive_main_moves"]["last"] == pytest.approx(2.0)
    assert actions["max_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert any("collector_max_consecutive_main_moves exceeded" in warning for warning in summary["warnings"])


def test_learning_progress_diagnostic_warns_when_latest_alias_mismatches_tracker_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best.pt").write_bytes(b"best-weights")
    (checkpoint_dir / "latest.pt").write_bytes(b"stale-latest-weights")
    (checkpoint_dir / "observed_best.pt").write_bytes(b"stale-observed-weights")
    (checkpoint_dir / "checkpoint_tracker.json").write_text(
        json.dumps(
            {
                "format": "checkpoint_tracker_v1",
                "latest": {
                    "alias": "latest",
                    "alias_path": "training/checkpoints/latest.pt",
                    "source_checkpoint_path": "training/checkpoints/best.pt",
                },
                "best": {
                    "alias": "best",
                    "alias_path": "training/checkpoints/best.pt",
                    "source_checkpoint_path": "training/checkpoints/checkpoint_50.pt",
                    "metric_kind": "dev_eval_mean",
                    "metric_value": 0.5,
                },
                "observed_best": {
                    "alias": "observed_best",
                    "alias_path": "training/checkpoints/observed_best.pt",
                    "source_checkpoint_path": "training/checkpoints/best.pt",
                    "metric_kind": "dev_eval_observed_mean",
                    "metric_value": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_learning_progress_summary(run_dir)

    assert summary["checkpoint_alias_integrity"]["latest_alias_matches_source"] is False
    assert summary["checkpoint_alias_integrity"]["observed_best_alias_matches_source"] is False
    assert any("latest checkpoint alias file does not match" in warning for warning in summary["warnings"])
    assert any("observed_best checkpoint alias file does not match" in warning for warning in summary["warnings"])


def test_learning_progress_diagnostic_summarizes_generic_final_eval_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_tracker.json").write_text(
        json.dumps(
            {
                "format": "checkpoint_tracker_v1",
                "best": {
                    "alias": "best",
                    "metric_kind": "dev_eval_mean",
                    "metric_value": 0.56,
                    "policy_version": 2,
                    "source_checkpoint_path": "training/checkpoints/checkpoint_50.pt",
                    "update_count": 50,
                },
            }
        ),
        encoding="utf-8",
    )
    snapshot_dir = run_dir / "training" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.joinpath("registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshots": [
                    {"policy_id": "policy_000002", "update": 50, "path": "snapshots/p2.pt"},
                    {"policy_id": "policy_000003", "update": 75, "path": "snapshots/p3.pt"},
                    {"policy_id": "policy_000004", "update": 100, "path": "snapshots/p4.pt"},
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix_dir = run_dir / "eval" / "final_eval" / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.joinpath("mean.csv").write_text(
        "\n".join(
            (
                "focal_policy_id,policy_000002,policy_000003,policy_000004",
                "policy_000002,0.5,0.52,0.45",
                "policy_000003,0.48,0.5,0.51",
                "policy_000004,0.56,0.53,0.5",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_dir.joinpath("wins.csv").write_text(
        "\n".join(
            (
                "focal_policy_id,policy_000002,policy_000003,policy_000004",
                "policy_000002,32,33,29",
                "policy_000003,31,32,33",
                "policy_000004,36,34,32",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    archive_matrix_dir = run_dir / "eval" / "final_eval_argmax_probe" / "matrices"
    archive_matrix_dir.mkdir(parents=True, exist_ok=True)
    archive_matrix_dir.joinpath("mean.csv").write_text(
        "\n".join(
            (
                "focal_policy_id,policy_000002,policy_000003,policy_000004",
                "policy_000002,0.5,0.70,0.60",
                "policy_000003,0.30,0.5,0.55",
                "policy_000004,0.40,0.45,0.5",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    archive_matrix_dir.joinpath("wins.csv").write_text(
        "\n".join(
            (
                "focal_policy_id,policy_000002,policy_000003,policy_000004",
                "policy_000002,32,45,38",
                "policy_000003,19,32,35",
                "policy_000004,26,29,32",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_learning_progress_summary(run_dir)

    matrix = summary["final_eval_matrix"]
    assert matrix["mean"]["row_policy_ids"] == ["policy_000002", "policy_000003", "policy_000004"]
    assert matrix["wins"]["values"][2] == [36.0, 34.0, 32.0]
    assert matrix["checkpoint_best_policy_id"] == "policy_000002"
    assert matrix["checkpoint_best_row_mean_excluding_self"] == pytest.approx(0.485)
    assert matrix["best_row_policy_id"] == "policy_000004"
    assert matrix["best_row_update"] == 100
    assert matrix["best_row_mean_excluding_self"] == pytest.approx(0.545)
    matrices = summary["final_eval_matrices"]
    assert sorted(matrices) == ["final_eval", "final_eval_argmax_probe"]
    assert matrices["final_eval_argmax_probe"]["checkpoint_best_policy_id"] == "policy_000002"
    assert matrices["final_eval_argmax_probe"]["checkpoint_best_row_mean_excluding_self"] == pytest.approx(0.65)
    assert matrices["final_eval_argmax_probe"]["wins"]["values"][0] == [32.0, 45.0, 38.0]
    assert any("not the strongest row" in warning for warning in summary["warnings"])
