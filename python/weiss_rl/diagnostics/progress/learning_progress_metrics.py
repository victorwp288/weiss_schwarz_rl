from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.progress import learning_progress_math as _progress_math
from weiss_rl.diagnostics.progress.learning_progress_sync import (
    build_actor_model_sync_section,
    build_actor_sync_series,
    build_league_sync_section,
    build_off_policy_section,
    build_off_policy_series,
)
from weiss_rl.diagnostics.progress.learning_progress_teacher_guidance import build_teacher_guidance_series
from weiss_rl.diagnostics.progress.learning_progress_warnings import (
    TrainingLogWarningInputs,
    build_training_log_warnings,
)

# Compatibility exports: generic math lives in learning_progress_math, while
# this module owns the domain-specific training-log summary.
_fraction_values = _progress_math._fraction_values
_last_window_mean = _progress_math._last_window_mean
_mean = _progress_math._mean
_numeric_by_update = _progress_math._numeric_by_update
_numeric_value = _progress_math._numeric_value
_numeric_values = _progress_math._numeric_values
_paired_update_values = _progress_math._paired_update_values
_pearson_correlation = _progress_math._pearson_correlation
_ratio_values = _progress_math._ratio_values
_sum_fraction_values = _progress_math._sum_fraction_values
_window_summary = _progress_math._window_summary

@dataclass(frozen=True)
class TrainingLogSummarySections:
    sections: dict[str, Any]
    warnings: list[str]


def build_training_log_summary_sections(
    *,
    metrics: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    promotion_gate: Mapping[str, Any],
) -> TrainingLogSummarySections:
    records_for_route = scalars + performance
    records_for_learning = metrics + scalars
    actor_heuristic_values = _numeric_values(records_for_route, "actor_heuristic_fraction_active")
    heuristic_mix_values = _numeric_values(records_for_route, "heuristic_public_mix_fraction_active")
    pfsp_pool_size_values = _numeric_values(records_for_route, "pfsp_pool_size")
    pfsp_champion_pool_size_values = _numeric_values(records_for_route, "pfsp_champion_pool_size")
    pfsp_recent_pool_size_values = _numeric_values(records_for_route, "pfsp_recent_pool_size")
    pfsp_hard_negative_pool_size_values = _numeric_values(records_for_route, "pfsp_hard_negative_pool_size")
    pfsp_quarantined_opponent_values = _numeric_values(records_for_route, "pfsp_quarantined_opponents")
    pfsp_snapshot_env_fraction_values = _sum_fraction_values(
        records_for_route,
        (
            "pfsp_champion_envs",
            "pfsp_recent_envs",
            "pfsp_hard_negative_envs",
            "pfsp_warmup_snapshot_envs",
        ),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_recent_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_recent_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_champion_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_champion_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_hard_negative_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_hard_negative_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_warmup_snapshot_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_warmup_snapshot_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    actor_sync = build_actor_sync_series(records_for_route=records_for_route, metrics=metrics)
    off_policy = build_off_policy_series(metrics)
    reward_mean_values = _numeric_values(metrics, "reward_mean")
    reward_abs_values = _numeric_values(metrics, "reward_abs_mean")
    reward_std_values = _numeric_values(metrics, "reward_std")
    reward_nonzero_values = _numeric_values(metrics, "reward_nonzero_fraction")
    reward_positive_values = _numeric_values(metrics, "reward_positive_fraction")
    reward_negative_values = _numeric_values(metrics, "reward_negative_fraction")
    advantage_abs_values = _numeric_values(metrics, "advantage_abs_mean")
    target_abs_values = _numeric_values(metrics, "target_abs_mean")
    chosen_pass_train_fraction_values = _numeric_values(metrics, "chosen_pass_train_fraction")
    chosen_pass_train_advantage_values = _numeric_values(metrics, "chosen_pass_train_advantage_mean")
    chosen_nonpass_train_advantage_values = _numeric_values(metrics, "chosen_nonpass_train_advantage_mean")
    chosen_mulligan_confirm_train_fraction_values = _numeric_values(
        metrics,
        "chosen_mulligan_confirm_train_fraction",
    )
    chosen_mulligan_select_train_fraction_values = _numeric_values(metrics, "chosen_mulligan_select_train_fraction")
    chosen_mulligan_confirm_train_advantage_values = _numeric_values(
        metrics,
        "chosen_mulligan_confirm_train_advantage_mean",
    )
    chosen_mulligan_select_train_advantage_values = _numeric_values(
        metrics,
        "chosen_mulligan_select_train_advantage_mean",
    )
    chosen_mulligan_select_share_values = _ratio_values(
        metrics,
        "chosen_mulligan_select_train_fraction",
        ("chosen_mulligan_select_train_fraction", "chosen_mulligan_confirm_train_fraction"),
    )
    chosen_play_train_fraction_values = _numeric_values(metrics, "chosen_main_play_character_train_fraction")
    chosen_attack_train_fraction_values = _numeric_values(metrics, "chosen_attack_train_fraction")
    teacher_guidance = build_teacher_guidance_series(records_for_learning=records_for_learning, scalars=scalars)
    main_move_fraction_values = _fraction_values(scalars, "collector_main_move_actions", "collector_total_actions")
    pass_fraction_values = _fraction_values(scalars, "collector_pass_actions", "collector_total_actions")
    pass_with_nonpass_total_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_available",
        "collector_total_actions",
    )
    pass_with_nonpass_pass_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_available",
        "collector_pass_actions",
    )
    pass_penalty_total_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_penalty_count",
        "collector_total_actions",
    )
    pass_penalty_pass_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_penalty_count",
        "collector_pass_actions",
    )
    mulligan_penalty_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_select_with_confirm_penalty_count",
        "collector_total_actions",
    )
    mulligan_guard_rows_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_force_confirm_after_select_rows",
        "collector_total_actions",
    )
    mulligan_guard_actions_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_force_confirm_after_select_actions",
        "collector_total_actions",
    )
    main_move_guard_rows_total_fraction_values = _fraction_values(
        scalars,
        "collector_main_move_only_force_pass_rows",
        "collector_total_actions",
    )
    main_move_guard_actions_total_fraction_values = _fraction_values(
        scalars,
        "collector_main_move_only_force_pass_actions",
        "collector_total_actions",
    )
    max_consecutive_main_move_values = _numeric_values(scalars, "collector_max_consecutive_main_moves")

    latest_champion_pool_size = None if not pfsp_champion_pool_size_values else pfsp_champion_pool_size_values[-1]
    latest_snapshot_env_fraction = (
        None if not pfsp_snapshot_env_fraction_values else pfsp_snapshot_env_fraction_values[-1]
    )
    latest_recent_env_fraction = None if not pfsp_recent_env_fraction_values else pfsp_recent_env_fraction_values[-1]
    warnings = build_training_log_warnings(
        TrainingLogWarningInputs(
            actor_heuristic_values=actor_heuristic_values,
            heuristic_mix_values=heuristic_mix_values,
            actor_lag_warning_values=actor_sync.lag_warning_values,
            actor_lag_warning_source=actor_sync.lag_warning_source,
            latest_champion_pool_size=latest_champion_pool_size,
            latest_snapshot_env_fraction=latest_snapshot_env_fraction,
            promotion_gate=promotion_gate,
            vtrace_rho_values=off_policy.vtrace_rho_values,
            vtrace_rho_p99_values=off_policy.vtrace_rho_p99_values,
            vtrace_train_rho_values=off_policy.vtrace_train_rho_values,
            vtrace_train_rho_p95_values=off_policy.vtrace_train_rho_p95_values,
            vtrace_train_rho_p99_values=off_policy.vtrace_train_rho_p99_values,
            vtrace_clip_rate_values=off_policy.vtrace_clip_rate_values,
            train_logp_delta_abs_p99_values=off_policy.train_logp_delta_abs_p99_values,
            max_consecutive_main_move_values=max_consecutive_main_move_values,
            chosen_pass_train_fraction_values=chosen_pass_train_fraction_values,
            chosen_mulligan_select_share_values=chosen_mulligan_select_share_values,
            pass_with_nonpass_total_fraction_values=pass_with_nonpass_total_fraction_values,
            teacher_public_heuristic_coef_active_values=(
                teacher_guidance.teacher_public_heuristic_coef_active_values
            ),
            teacher_public_heuristic_supported_values=(
                teacher_guidance.teacher_public_heuristic_supported_values
            ),
            teacher_hand_coef_active_values=teacher_guidance.teacher_hand_coef_active_values,
            teacher_hand_supported_values=teacher_guidance.teacher_hand_supported_values,
        )
    )

    sections = {
        "loss": _window_summary(_numeric_values(metrics, "loss"), window=20),
        "teacher_family_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_family_accuracy"), window=20
        ),
        "teacher_slot_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_slot_accuracy"), window=20
        ),
        "teacher_action_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_action_accuracy"), window=20
        ),
        "teacher_guidance": teacher_guidance.section(),
        "route": {
            "max_actor_heuristic_fraction_active": None if not actor_heuristic_values else max(actor_heuristic_values),
            "max_heuristic_public_mix_fraction_active": None if not heuristic_mix_values else max(heuristic_mix_values),
        },
        "league_sampling": {
            "pfsp_pool_size": _window_summary(pfsp_pool_size_values, window=20),
            "pfsp_champion_pool_size": _window_summary(pfsp_champion_pool_size_values, window=20),
            "pfsp_recent_pool_size": _window_summary(pfsp_recent_pool_size_values, window=20),
            "pfsp_hard_negative_pool_size": _window_summary(pfsp_hard_negative_pool_size_values, window=20),
            "pfsp_quarantined_opponents": _window_summary(pfsp_quarantined_opponent_values, window=20),
            "snapshot_env_fraction": _window_summary(pfsp_snapshot_env_fraction_values, window=20),
            "champion_env_fraction": _window_summary(pfsp_champion_env_fraction_values, window=20),
            "recent_env_fraction": _window_summary(pfsp_recent_env_fraction_values, window=20),
            "hard_negative_env_fraction": _window_summary(pfsp_hard_negative_env_fraction_values, window=20),
            "warmup_snapshot_env_fraction": _window_summary(pfsp_warmup_snapshot_env_fraction_values, window=20),
            "max_snapshot_env_fraction": None
            if not pfsp_snapshot_env_fraction_values
            else max(pfsp_snapshot_env_fraction_values),
            "latest_has_admitted_champion": bool(latest_champion_pool_size and latest_champion_pool_size > 0.0),
            "latest_probationary_recent_sampling_active": bool(
                latest_champion_pool_size == 0.0
                and latest_recent_env_fraction is not None
                and latest_recent_env_fraction > 0.0
            ),
        },
        "actor_model_sync": build_actor_model_sync_section(actor_sync),
        "league_sync": build_league_sync_section(actor_sync),
        "off_policy": build_off_policy_section(actor_sync=actor_sync, off_policy=off_policy),
        "reward_scale": {
            "reward_mean": _window_summary(reward_mean_values, window=20),
            "reward_abs_mean": _window_summary(reward_abs_values, window=20),
            "reward_std": _window_summary(reward_std_values, window=20),
            "reward_nonzero_fraction": _window_summary(reward_nonzero_values, window=20),
            "reward_positive_fraction": _window_summary(reward_positive_values, window=20),
            "reward_negative_fraction": _window_summary(reward_negative_values, window=20),
            "advantage_abs_mean": _window_summary(advantage_abs_values, window=20),
            "target_abs_mean": _window_summary(target_abs_values, window=20),
            "max_reward_abs_mean": None if not reward_abs_values else max(reward_abs_values),
            "max_target_abs_mean": None if not target_abs_values else max(target_abs_values),
        },
        "chosen_action_learning": {
            "chosen_pass_train_fraction": _window_summary(chosen_pass_train_fraction_values, window=20),
            "chosen_pass_train_advantage_mean": _window_summary(chosen_pass_train_advantage_values, window=20),
            "chosen_nonpass_train_advantage_mean": _window_summary(chosen_nonpass_train_advantage_values, window=20),
            "chosen_mulligan_confirm_train_fraction": _window_summary(
                chosen_mulligan_confirm_train_fraction_values,
                window=20,
            ),
            "chosen_mulligan_select_train_fraction": _window_summary(
                chosen_mulligan_select_train_fraction_values,
                window=20,
            ),
            "chosen_mulligan_select_share_of_mulligan": _window_summary(
                chosen_mulligan_select_share_values,
                window=20,
            ),
            "chosen_mulligan_confirm_train_advantage_mean": _window_summary(
                chosen_mulligan_confirm_train_advantage_values,
                window=20,
            ),
            "chosen_mulligan_select_train_advantage_mean": _window_summary(
                chosen_mulligan_select_train_advantage_values,
                window=20,
            ),
            "chosen_main_play_character_train_fraction": _window_summary(chosen_play_train_fraction_values, window=20),
            "chosen_attack_train_fraction": _window_summary(chosen_attack_train_fraction_values, window=20),
        },
        "action_distribution": {
            "main_move_fraction": _window_summary(main_move_fraction_values, window=20),
            "pass_fraction": _window_summary(pass_fraction_values, window=20),
            "pass_with_nonpass_fraction_of_total": _window_summary(
                pass_with_nonpass_total_fraction_values,
                window=20,
            ),
            "pass_with_nonpass_fraction_of_pass": _window_summary(
                pass_with_nonpass_pass_fraction_values,
                window=20,
            ),
            "pass_penalty_fraction_of_total": _window_summary(pass_penalty_total_fraction_values, window=20),
            "pass_penalty_fraction_of_pass": _window_summary(pass_penalty_pass_fraction_values, window=20),
            "mulligan_select_with_confirm_penalty_fraction_of_total": _window_summary(
                mulligan_penalty_total_fraction_values,
                window=20,
            ),
            "mulligan_force_confirm_after_select_rows_fraction_of_total": _window_summary(
                mulligan_guard_rows_total_fraction_values,
                window=20,
            ),
            "mulligan_force_confirm_after_select_actions_fraction_of_total": _window_summary(
                mulligan_guard_actions_total_fraction_values,
                window=20,
            ),
            "main_move_only_force_pass_rows_fraction_of_total": _window_summary(
                main_move_guard_rows_total_fraction_values,
                window=20,
            ),
            "main_move_only_force_pass_actions_fraction_of_total": _window_summary(
                main_move_guard_actions_total_fraction_values,
                window=20,
            ),
            "max_consecutive_main_moves": _window_summary(max_consecutive_main_move_values, window=20),
            "max_max_consecutive_main_moves": None
            if not max_consecutive_main_move_values
            else max(max_consecutive_main_move_values),
        },
    }
    return TrainingLogSummarySections(sections=sections, warnings=warnings)
