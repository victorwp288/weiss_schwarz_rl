from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weiss_rl.diagnostics.progress.learning_progress import build_learning_progress_summary


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_overheated_training_summary(tmp_path: Path) -> dict[str, Any]:
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
    return build_learning_progress_summary(run_dir)


def build_action_distribution_summary(tmp_path: Path) -> dict[str, Any]:
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
    return build_learning_progress_summary(run_dir)


def build_stale_checkpoint_alias_summary(tmp_path: Path) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best.pt").write_bytes(b"best-weights")
    (checkpoint_dir / "latest.pt").write_bytes(b"stale-latest-weights")
    (checkpoint_dir / "observed_best.pt").write_bytes(b"stale-observed-weights")
    _write_json(
        checkpoint_dir / "checkpoint_tracker.json",
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
        },
    )
    return build_learning_progress_summary(run_dir)


def write_periodic_dev_eval_trend_fixture(run_dir: Path) -> None:
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
    _write_json(
        run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
        {
            "train_u25_p1": {"update_count": 25, "aggregate_score": 0.45, "anchor_scores": {"B0": 0.8}},
            "train_u50_p2": {"update_count": 50, "aggregate_score": 0.60, "anchor_scores": {"B0": 0.9}},
            "train_u75_p3": {"update_count": 75, "aggregate_score": 0.52, "anchor_scores": {"B0": 0.85}},
        },
    )


def write_promotion_gate_failure_fixture(run_dir: Path) -> None:
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


def write_final_eval_matrix_fixture(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        checkpoint_dir / "checkpoint_tracker.json",
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
        },
    )
    snapshot_dir = run_dir / "training" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        snapshot_dir / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {"policy_id": "policy_000002", "update": 50, "path": "snapshots/p2.pt"},
                {"policy_id": "policy_000003", "update": 75, "path": "snapshots/p3.pt"},
                {"policy_id": "policy_000004", "update": 100, "path": "snapshots/p4.pt"},
            ],
        },
    )
    _write_matrix_csv(
        run_dir / "eval" / "final_eval" / "matrices",
        mean_rows=[
            "policy_000002,0.5,0.52,0.45",
            "policy_000003,0.48,0.5,0.51",
            "policy_000004,0.56,0.53,0.5",
        ],
        win_rows=[
            "policy_000002,32,33,29",
            "policy_000003,31,32,33",
            "policy_000004,36,34,32",
        ],
    )
    _write_matrix_csv(
        run_dir / "eval" / "final_eval_argmax_probe" / "matrices",
        mean_rows=[
            "policy_000002,0.5,0.70,0.60",
            "policy_000003,0.30,0.5,0.55",
            "policy_000004,0.40,0.45,0.5",
        ],
        win_rows=[
            "policy_000002,32,45,38",
            "policy_000003,19,32,35",
            "policy_000004,26,29,32",
        ],
    )


def _write_matrix_csv(matrix_dir: Path, *, mean_rows: list[str], win_rows: list[str]) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    header = "focal_policy_id,policy_000002,policy_000003,policy_000004"
    matrix_dir.joinpath("mean.csv").write_text("\n".join([header, *mean_rows]) + "\n", encoding="utf-8")
    matrix_dir.joinpath("wins.csv").write_text("\n".join([header, *win_rows]) + "\n", encoding="utf-8")
