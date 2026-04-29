"""Metric projection helpers for learner logging."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from weiss_rl.learners.vtrace import VtraceMetrics

CUSTOM_LOG_METRIC_KEYS = (
    "policy_loss_coef",
    "behavior_action_bc_loss",
    "behavior_action_bc_coef",
    "reference_policy_top_action_bc_loss",
    "reference_policy_top_action_bc_coef",
    "reference_policy_top_action_family_bc_loss",
    "reference_policy_top_action_family_bc_coef",
    "raw_b1_distill_loss",
    "raw_b1_distill_coef",
    "raw_b1_distill_teacher_bias_scale",
    "raw_b1_distill_student_bias_scale",
    "raw_b1_distill_temperature",
    "raw_b1_distill_top_k",
    "raw_b1_distill_top_action_ce_coef",
    "raw_b1_distill_row_fraction",
    "raw_b1_top1_match",
    "raw_b1_topk_overlap",
    "raw_b1_family_match",
    "raw_b1_kl",
    "raw_b1_top_action_ce",
    "counterfactual_positive_loss",
    "counterfactual_positive_coef",
    "counterfactual_positive_margin_coef",
    "counterfactual_positive_margin",
    "counterfactual_positive_ce_loss",
    "counterfactual_positive_margin_loss",
    "counterfactual_positive_label_count",
    "counterfactual_positive_weight_mean",
    "counterfactual_positive_prob_mean",
    "counterfactual_positive_top1_match",
    "counterfactual_positive_logit_margin_mean",
    "b1_opponent_reference_policy_top_action_bc_loss",
    "b1_opponent_reference_policy_top_action_bc_coef",
    "b1_opponent_reference_policy_top_action_bc_row_fraction",
    "b1_second_seat_positive_advantage_policy_loss",
    "b1_second_seat_positive_advantage_policy_coef",
    "b1_second_seat_positive_advantage_row_fraction",
    "b1_second_seat_positive_advantage_mean",
    "b1_second_seat_reference_top_action_avoidance_loss",
    "b1_second_seat_reference_top_action_avoidance_coef",
    "b1_second_seat_reference_top_action_avoidance_row_fraction",
    "policy_train_fraction",
    "reward_mean",
    "reward_abs_mean",
    "reward_nonzero_fraction",
    "advantage_mean",
    "advantage_abs_mean",
    "target_mean",
    "target_abs_mean",
    "teacher_development_pass_suppression_loss",
    "teacher_development_pass_suppression_selected_fraction",
    "teacher_development_pass_probability",
)


def build_custom_log_metrics(
    update_metrics: Mapping[str, float],
    vtrace_metrics: VtraceMetrics,
) -> dict[str, float]:
    custom_metrics: dict[str, float] = {
        "vtrace_batch_metrics_available": float(np.isfinite(vtrace_metrics.rho_mean)),
    }
    if "vtrace_rho_p95" in update_metrics:
        custom_metrics["vtrace_rho_p95"] = float(update_metrics["vtrace_rho_p95"])
    if np.isfinite(vtrace_metrics.entropy):
        custom_metrics["vtrace_entropy"] = float(vtrace_metrics.entropy)
    for key in CUSTOM_LOG_METRIC_KEYS:
        if key in update_metrics:
            custom_metrics[key] = float(update_metrics[key])
    return custom_metrics
