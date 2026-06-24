from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.progress.learning_progress_math import _last_window_mean

_OFF_POLICY_RHO_WARN_THRESHOLD = 10.0
_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD = 10.0
_VTRACE_CLIP_RATE_WARN_THRESHOLD = 0.5
_LEARNER_ACTOR_LAG_WARN_THRESHOLD = 25.0
_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD = 1.0
_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD = 1.0
_MULLIGAN_SELECT_SHARE_WARN_THRESHOLD = 0.8
_TEACHER_SUPPORTED_WARN_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class TrainingLogWarningInputs:
    actor_heuristic_values: list[float]
    heuristic_mix_values: list[float]
    actor_lag_warning_values: list[float]
    actor_lag_warning_source: str
    latest_champion_pool_size: float | None
    latest_snapshot_env_fraction: float | None
    promotion_gate: Mapping[str, Any]
    vtrace_rho_values: list[float]
    vtrace_rho_p99_values: list[float]
    vtrace_train_rho_values: list[float]
    vtrace_train_rho_p95_values: list[float]
    vtrace_train_rho_p99_values: list[float]
    vtrace_clip_rate_values: list[float]
    train_logp_delta_abs_p99_values: list[float]
    max_consecutive_main_move_values: list[float]
    chosen_pass_train_fraction_values: list[float]
    chosen_mulligan_select_share_values: list[float]
    pass_with_nonpass_total_fraction_values: list[float]
    teacher_public_heuristic_coef_active_values: list[float]
    teacher_public_heuristic_supported_values: list[float]
    teacher_hand_coef_active_values: list[float]
    teacher_hand_supported_values: list[float]


def build_training_log_warnings(inputs: TrainingLogWarningInputs) -> list[str]:
    warnings: list[str] = []
    if inputs.actor_heuristic_values and max(inputs.actor_heuristic_values) > 0.0:
        warnings.append("actor_heuristic_fraction_active was nonzero; focal actions were not pure model-policy rows")
    if inputs.heuristic_mix_values and max(inputs.heuristic_mix_values) > 0.0:
        warnings.append("heuristic_public_mix_fraction_active was nonzero; eval/train pressure includes B2 heuristic")
    if (
        inputs.actor_lag_warning_values
        and max(inputs.actor_lag_warning_values) > _LEARNER_ACTOR_LAG_WARN_THRESHOLD
    ):
        warnings.append(
            f"{inputs.actor_lag_warning_source} exceeded "
            f"{_LEARNER_ACTOR_LAG_WARN_THRESHOLD:g}; actor policy may be stale"
        )

    if inputs.promotion_gate["attempt_count"] > 0 and inputs.promotion_gate["passed_count"] == 0:
        if inputs.latest_champion_pool_size is not None and inputs.latest_champion_pool_size > 0.0:
            warnings.append(
                "promotion gate never passed; champion pool is populated by imported/bootstrap champions, "
                "not promoted trained champions"
            )
        elif (
            inputs.latest_champion_pool_size == 0.0
            and inputs.latest_snapshot_env_fraction is not None
            and inputs.latest_snapshot_env_fraction > 0.0
        ):
            warnings.append(
                "promotion gate never passed; no trained champions were admitted, but probationary snapshot "
                "sampling was active"
            )
        else:
            warnings.append("promotion gate never passed; league did not admit any trained champions")
    if int(inputs.promotion_gate["consecutive_failure_count"]) >= 3:
        warnings.append(
            "promotion gate failed "
            f"{int(inputs.promotion_gate['consecutive_failure_count'])} consecutive attempts through latest update"
        )

    warnings.extend(_off_policy_warnings(inputs))
    warnings.extend(_action_distribution_warnings(inputs))
    warnings.extend(_teacher_support_warnings(inputs))
    return warnings


def _off_policy_warnings(inputs: TrainingLogWarningInputs) -> list[str]:
    warnings: list[str] = []
    if inputs.vtrace_rho_values and max(inputs.vtrace_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; behavior/evaluation log-probs may be mismatched"
        )
    train_rho_tail_values = inputs.vtrace_train_rho_p99_values or inputs.vtrace_train_rho_p95_values
    max_train_rho_tail = None if not train_rho_tail_values else max(train_rho_tail_values)
    if inputs.vtrace_rho_p99_values and max(inputs.vtrace_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        if max_train_rho_tail is not None and max_train_rho_tail <= _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
            warnings.append(
                "raw vtrace_rho_p99 exceeded "
                f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}, but train-mask rho tail stayed below threshold; "
                "large off-policy tails are mostly filtered or non-train rows"
            )
        else:
            warnings.append(
                "vtrace_rho_p99 exceeded "
                f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; off-policy correction tails are large"
            )
    if inputs.vtrace_train_rho_values and max(inputs.vtrace_train_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; train-mask behavior/evaluation log-probs may be mismatched"
        )
    if (
        inputs.vtrace_train_rho_p95_values
        and max(inputs.vtrace_train_rho_p95_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD
    ):
        warnings.append(
            "vtrace_train_rho_p95 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if (
        inputs.vtrace_train_rho_p99_values
        and max(inputs.vtrace_train_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD
    ):
        warnings.append(
            "vtrace_train_rho_p99 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if inputs.vtrace_clip_rate_values and max(inputs.vtrace_clip_rate_values) > _VTRACE_CLIP_RATE_WARN_THRESHOLD:
        warnings.append(
            f"vtrace_clip_rate exceeded {_VTRACE_CLIP_RATE_WARN_THRESHOLD:g}; policy updates are heavily clipped"
        )
    if (
        inputs.train_logp_delta_abs_p99_values
        and max(inputs.train_logp_delta_abs_p99_values) > _TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD
    ):
        warnings.append(
            "target_behavior_train_logp_delta_abs_p99 exceeded "
            f"{_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD:g}; learner and behavior log-probs diverged on train rows"
        )
    return warnings


def _action_distribution_warnings(inputs: TrainingLogWarningInputs) -> list[str]:
    warnings: list[str] = []
    if (
        inputs.max_consecutive_main_move_values
        and max(inputs.max_consecutive_main_move_values) > _MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD
    ):
        warnings.append(
            "collector_max_consecutive_main_moves exceeded "
            f"{_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD:g}; repeated main-move transitions or counter drift suspected"
        )
    chosen_pass_train_fraction_last = _last_window_mean(inputs.chosen_pass_train_fraction_values, window=20)
    if chosen_pass_train_fraction_last is not None and chosen_pass_train_fraction_last > 0.5:
        warnings.append("chosen_pass_train_fraction averaged above 0.5 in the latest window; pass-collapse suspected")
    chosen_mulligan_select_share_last = _last_window_mean(inputs.chosen_mulligan_select_share_values, window=20)
    if (
        chosen_mulligan_select_share_last is not None
        and chosen_mulligan_select_share_last > _MULLIGAN_SELECT_SHARE_WARN_THRESHOLD
    ):
        warnings.append(
            "chosen_mulligan_select share among mulligan actions is high in the latest window; "
            "mulligan-confirm collapse suspected"
        )
    pass_with_nonpass_total_fraction_last = _last_window_mean(inputs.pass_with_nonpass_total_fraction_values, window=20)
    if pass_with_nonpass_total_fraction_last is not None and pass_with_nonpass_total_fraction_last > 0.35:
        warnings.append(
            "collector pass-with-nonpass fraction is high in the latest window; policy may be avoiding play"
        )
    return warnings


def _teacher_support_warnings(inputs: TrainingLogWarningInputs) -> list[str]:
    warnings: list[str] = []
    teacher_public_heuristic_coef_active_last = _last_window_mean(
        inputs.teacher_public_heuristic_coef_active_values, window=20
    )
    teacher_public_heuristic_supported_last = _last_window_mean(
        inputs.teacher_public_heuristic_supported_values, window=20
    )
    if (
        teacher_public_heuristic_coef_active_last is not None
        and teacher_public_heuristic_coef_active_last > 0.0
        and (
            teacher_public_heuristic_supported_last is None
            or teacher_public_heuristic_supported_last < _TEACHER_SUPPORTED_WARN_THRESHOLD
        )
    ):
        warnings.append(
            "teacher_public_heuristic_coef_active was nonzero but public-teacher support was near zero; "
            "teacher labels or packed metadata may be missing"
        )
    teacher_hand_coef_active_last = _last_window_mean(inputs.teacher_hand_coef_active_values, window=20)
    teacher_hand_supported_last = _last_window_mean(inputs.teacher_hand_supported_values, window=20)
    if (
        teacher_hand_coef_active_last is not None
        and teacher_hand_coef_active_last > 0.0
        and (teacher_hand_supported_last is None or teacher_hand_supported_last < _TEACHER_SUPPORTED_WARN_THRESHOLD)
    ):
        warnings.append(
            "teacher_hand_coef_active was nonzero but hand-target support was near zero; "
            "hand metadata or factorized same-family arg0 references may be missing"
        )
    return warnings
