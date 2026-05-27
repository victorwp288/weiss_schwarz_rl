"""Checkpoint metadata and metric logging helpers for learners."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weiss_rl.diagnostics.training_logger import TrainingMetrics
from weiss_rl.learners.vtrace import VtraceMetrics


def checkpoint_metadata_payload(*, update_count: int, policy_version: int) -> dict[str, int | bool | str]:
    """Build the learner checkpoint metadata sidecar payload."""
    return {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "update_count": update_count,
        "policy_version": policy_version,
    }


def write_checkpoint_metadata(*, checkpoint_dir: Path | None, update_count: int, policy_version: int) -> Path | None:
    """Write a checkpoint metadata sidecar and return its path when enabled."""
    if not checkpoint_dir:
        return None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata_path = checkpoint_dir / f"checkpoint_metadata_{update_count}.json"
    checkpoint_metadata_path.write_text(
        json.dumps(
            checkpoint_metadata_payload(update_count=update_count, policy_version=policy_version),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint_metadata_path


def custom_log_metrics(update_metrics: dict[str, float], vtrace_metrics: VtraceMetrics) -> dict[str, float]:
    """Build learner custom metrics appended to training logs."""
    custom_metrics: dict[str, float] = {
        "vtrace_batch_metrics_available": float(np.isfinite(vtrace_metrics.rho_mean)),
    }
    for key in (
        "vtrace_rho_mean",
        "vtrace_rho_p95",
        "vtrace_train_rho_mean",
        "vtrace_train_rho_p95",
        "target_behavior_logp_delta_abs_mean",
        "target_behavior_logp_delta_abs_p95",
        "target_behavior_logp_delta_abs_p99",
        "target_behavior_train_logp_delta_abs_mean",
        "target_behavior_train_logp_delta_abs_p95",
        "target_behavior_train_logp_delta_abs_p99",
        "policy_train_fraction",
        "teacher_public_heuristic_coef_active",
        "teacher_supervised_coef_scale_active",
        "teacher_family_coef_active",
        "teacher_slot_coef_active",
        "teacher_hand_coef_active",
        "teacher_move_source_coef_active",
        "teacher_attack_type_coef_active",
        "teacher_action_coef_active",
        "teacher_same_family_action_coef_active",
        "teacher_action_margin_coef_active",
        "teacher_same_family_action_margin_coef_active",
        "teacher_label_profile_id_active",
        "teacher_label_profile_base_active",
        "teacher_label_profile_aggressive_active",
        "teacher_label_profile_control_active",
        "policy_anchor_coef_active",
        "policy_anchor_top_action_coef_active",
        "policy_anchor_temperature",
        "policy_anchor_loss",
        "policy_anchor_weighted_loss",
        "policy_anchor_kl_mean",
        "policy_anchor_kl_p95",
        "policy_anchor_candidate_count",
        "policy_anchor_top_action_loss",
        "policy_anchor_top_action_loss_p95",
        "policy_anchor_top_action_agreement",
        "trajectory_retention_coef_active",
        "trajectory_retention_valid_fraction",
        "trajectory_retention_supported_fraction",
        "trajectory_retention_rows",
        "trajectory_retention_loss",
        "trajectory_retention_weighted_loss",
        "trajectory_retention_logp_mean",
        "trajectory_retention_top_action_accuracy",
        "reward_mean",
        "reward_std",
        "reward_abs_mean",
        "reward_min",
        "reward_max",
        "reward_nonzero_fraction",
        "reward_positive_fraction",
        "reward_negative_fraction",
        "advantage_mean",
        "advantage_abs_mean",
        "target_mean",
        "target_abs_mean",
        "entropy_scope_family_active",
        "chosen_pass_train_fraction",
        "chosen_pass_train_reward_mean",
        "chosen_pass_train_advantage_mean",
        "chosen_nonpass_train_reward_mean",
        "chosen_nonpass_train_advantage_mean",
        "chosen_mulligan_confirm_train_fraction",
        "chosen_mulligan_confirm_train_reward_mean",
        "chosen_mulligan_confirm_train_advantage_mean",
        "chosen_mulligan_select_train_fraction",
        "chosen_mulligan_select_train_reward_mean",
        "chosen_mulligan_select_train_advantage_mean",
        "chosen_main_play_character_train_fraction",
        "chosen_main_play_character_train_reward_mean",
        "chosen_main_play_character_train_advantage_mean",
        "chosen_main_move_train_fraction",
        "chosen_main_move_train_reward_mean",
        "chosen_main_move_train_advantage_mean",
        "chosen_attack_train_fraction",
        "chosen_attack_train_reward_mean",
        "chosen_attack_train_advantage_mean",
        "teacher_family_accuracy",
        "teacher_slot_accuracy",
        "teacher_main_play_character_slot_accuracy",
        "teacher_hand_accuracy",
        "teacher_main_play_character_hand_accuracy",
        "teacher_clock_from_hand_accuracy",
        "teacher_hand_loss",
        "teacher_hand_supported_fraction",
        "teacher_attack_type_accuracy",
        "teacher_action_accuracy",
        "teacher_same_family_action_accuracy",
        "teacher_same_family_main_play_character_accuracy",
        "teacher_action_margin_loss",
        "teacher_action_margin_supported_fraction",
        "teacher_action_margin_mean",
        "teacher_action_margin_satisfied_fraction",
        "teacher_same_family_action_margin_loss",
        "teacher_same_family_action_margin_supported_fraction",
        "teacher_same_family_action_margin_mean",
        "teacher_same_family_action_margin_satisfied_fraction",
        "teacher_aux_loss",
        "teacher_public_heuristic_loss",
        "teacher_public_heuristic_supported_fraction",
        "teacher_public_heuristic_top1_mass",
        "teacher_public_heuristic_target_entropy",
        "teacher_public_nonpass_over_pass_loss",
        "teacher_public_nonpass_over_pass_supported_fraction",
        "teacher_public_nonpass_over_pass_margin_mean",
        "teacher_public_nonpass_over_pass_satisfied_fraction",
        "amp_grad_overflow",
    ):
        if key in update_metrics:
            custom_metrics[key] = float(update_metrics[key])
    for key, value in sorted(update_metrics.items()):
        if (key.startswith("trajectory_bc_replay_") or key.startswith("paired_swing_replay_")) and np.isfinite(
            float(value)
        ):
            custom_metrics[key] = float(value)
    if np.isfinite(vtrace_metrics.entropy):
        custom_metrics["vtrace_entropy"] = float(vtrace_metrics.entropy)
    return custom_metrics


def build_training_metrics(
    *,
    update_metrics: dict[str, float],
    vtrace_metrics: VtraceMetrics,
    update_count: int,
    policy_version: int,
    elapsed_seconds: float,
) -> TrainingMetrics:
    """Build the public training metrics record for one learner update."""
    return TrainingMetrics(
        update_count=update_count,
        wall_clock_seconds=elapsed_seconds,
        wall_clock_ms=int(elapsed_seconds * 1000),
        policy_version=policy_version,
        loss=float(update_metrics.get("loss", 0.0)),
        throughput_samples_per_sec=float(update_metrics.get("throughput_samples_per_sec", 0.0)),
        throughput_updates_per_sec=float(update_metrics.get("throughput_updates_per_sec", 0.0)),
        vtrace_rho_mean=float(update_metrics.get("vtrace_rho_mean", vtrace_metrics.rho_mean)),
        vtrace_rho_p50=float(update_metrics.get("vtrace_rho_p50", vtrace_metrics.rho_p50)),
        vtrace_rho_p90=float(update_metrics.get("vtrace_rho_p90", vtrace_metrics.rho_p90)),
        vtrace_rho_p99=float(update_metrics.get("vtrace_rho_p99", vtrace_metrics.rho_p99)),
        vtrace_clip_rate=float(update_metrics.get("vtrace_rho_clip_rate", vtrace_metrics.clip_rate)),
        vtrace_c_clipped_rate=float(update_metrics.get("vtrace_c_clip_rate", vtrace_metrics.c_clipped_rate)),
        kl_divergence=vtrace_metrics.kl_divergence,
        value_loss=float(update_metrics.get("value_loss", 0.0)),
        actor_loss=float(update_metrics.get("policy_loss", 0.0)),
        entropy=float(update_metrics.get("entropy", vtrace_metrics.entropy)),
        custom_metrics=custom_log_metrics(update_metrics, vtrace_metrics),
    )
