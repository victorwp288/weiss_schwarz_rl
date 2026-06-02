from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_OFF_POLICY_RHO_WARN_THRESHOLD = 10.0
_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD = 10.0
_VTRACE_CLIP_RATE_WARN_THRESHOLD = 0.5
_LEARNER_ACTOR_LAG_WARN_THRESHOLD = 25.0
_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD = 1.0
_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD = 1.0
_MULLIGAN_SELECT_SHARE_WARN_THRESHOLD = 0.8
_TEACHER_SUPPORTED_WARN_THRESHOLD = 0.05


@dataclass(frozen=True)
class TrainingLogSummarySections:
    sections: dict[str, Any]
    warnings: list[str]


def _numeric_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _numeric_value(record, key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _numeric_value(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if not isinstance(value, int | float):
        custom_metrics = record.get("custom_metrics")
        if isinstance(custom_metrics, dict):
            value = custom_metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _fraction_values(records: Iterable[dict[str, Any]], numerator_key: str, denominator_key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        denominator = _numeric_value(record, denominator_key)
        if numerator is None or denominator is None or denominator <= 0.0:
            continue
        values.append(float(numerator) / float(denominator))
    return values


def _ratio_values(
    records: Iterable[dict[str, Any]],
    numerator_key: str,
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        if numerator is None:
            continue
        denominator = 0.0
        complete = True
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if not complete or denominator <= 0.0:
            continue
        values.append(float(numerator) / denominator)
    return values


def _sum_fraction_values(
    records: Iterable[dict[str, Any]],
    numerator_keys: tuple[str, ...],
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = 0.0
        denominator = 0.0
        complete = True
        for key in numerator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            numerator += float(value)
        if not complete:
            continue
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if complete and denominator > 0.0:
            values.append(numerator / denominator)
    return values


def _numeric_by_update(records: Iterable[dict[str, Any]], key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        update_count = record.get("update_count")
        value = _numeric_value(record, key)
        if isinstance(update_count, int) and value is not None:
            values[int(update_count)] = float(value)
    return values


def _paired_update_values(
    left_records: Iterable[dict[str, Any]],
    left_key: str,
    right_records: Iterable[dict[str, Any]],
    right_key: str,
) -> list[tuple[float, float]]:
    left = _numeric_by_update(left_records, left_key)
    right = _numeric_by_update(right_records, right_key)
    return [(left[update], right[update]) for update in sorted(left.keys() & right.keys())]


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left_values = [left for left, _right in pairs]
    right_values = [right for _left, right in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    covariance = sum(left * right for left, right in zip(left_centered, right_centered, strict=True))
    return float(covariance / ((left_ss * right_ss) ** 0.5))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _window_summary(values: list[float], *, window: int) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None, "first_window_mean": None, "last_window_mean": None}
    first_window = values[:window]
    last_window = values[-window:]
    return {
        "first": values[0],
        "last": values[-1],
        "first_window_mean": _mean(first_window),
        "last_window_mean": _mean(last_window),
    }


def _last_window_mean(values: list[float], *, window: int) -> float | None:
    return _window_summary(values, window=window)["last_window_mean"]


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
    policy_version_lag_p50_values = _numeric_values(records_for_route, "policy_version_lag_p50")
    policy_version_lag_p90_values = _numeric_values(records_for_route, "policy_version_lag_p90")
    learner_actor_update_lag_p50_values = _numeric_values(records_for_route, "learner_actor_update_lag_p50")
    learner_actor_update_lag_p90_values = _numeric_values(records_for_route, "learner_actor_update_lag_p90")
    league_update_lag_values = _numeric_values(records_for_route, "league_update_lag")
    actor_lag_warning_values = (
        learner_actor_update_lag_p90_values or league_update_lag_values or policy_version_lag_p90_values
    )
    if learner_actor_update_lag_p90_values:
        actor_lag_warning_source = "learner_actor_update_lag_p90"
    elif league_update_lag_values:
        actor_lag_warning_source = "league_update_lag"
    else:
        actor_lag_warning_source = "policy_version_lag_p90"
    stale_policy_pairs = {
        "vtrace_rho_mean": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_rho_mean",
        ),
        "vtrace_rho_p99": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_rho_p99",
        ),
        "vtrace_train_rho_p95": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_train_rho_p95",
        ),
        "vtrace_train_rho_p99": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_train_rho_p99",
        ),
        "vtrace_clip_rate": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_clip_rate",
        ),
    }
    vtrace_rho_values = _numeric_values(metrics, "vtrace_rho_mean")
    vtrace_rho_p99_values = _numeric_values(metrics, "vtrace_rho_p99")
    vtrace_train_rho_values = _numeric_values(metrics, "vtrace_train_rho_mean")
    vtrace_train_rho_p95_values = _numeric_values(metrics, "vtrace_train_rho_p95")
    vtrace_train_rho_p99_values = _numeric_values(metrics, "vtrace_train_rho_p99")
    vtrace_clip_rate_values = _numeric_values(metrics, "vtrace_clip_rate")
    logp_delta_abs_values = _numeric_values(metrics, "target_behavior_logp_delta_abs_mean")
    logp_delta_abs_p99_values = _numeric_values(metrics, "target_behavior_logp_delta_abs_p99")
    train_logp_delta_abs_values = _numeric_values(metrics, "target_behavior_train_logp_delta_abs_mean")
    train_logp_delta_abs_p99_values = _numeric_values(metrics, "target_behavior_train_logp_delta_abs_p99")
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
    teacher_public_heuristic_coef_active_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_coef_active",
    )
    teacher_hand_coef_active_values = _numeric_values(records_for_learning, "teacher_hand_coef_active")
    teacher_aux_loss_values = _numeric_values(records_for_learning, "teacher_aux_loss")
    teacher_main_play_slot_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_main_play_character_slot_accuracy",
    )
    teacher_hand_accuracy_values = _numeric_values(records_for_learning, "teacher_hand_accuracy")
    teacher_main_play_hand_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_main_play_character_hand_accuracy",
    )
    teacher_clock_hand_accuracy_values = _numeric_values(records_for_learning, "teacher_clock_from_hand_accuracy")
    teacher_hand_loss_values = _numeric_values(records_for_learning, "teacher_hand_loss")
    teacher_hand_supported_values = _numeric_values(records_for_learning, "teacher_hand_supported_fraction")
    teacher_same_family_action_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_accuracy",
    )
    teacher_same_family_main_play_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_main_play_character_accuracy",
    )
    teacher_action_margin_mean_values = _numeric_values(records_for_learning, "teacher_action_margin_mean")
    teacher_action_margin_satisfied_values = _numeric_values(
        records_for_learning,
        "teacher_action_margin_satisfied_fraction",
    )
    teacher_same_family_action_margin_mean_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_margin_mean",
    )
    teacher_same_family_action_margin_satisfied_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_margin_satisfied_fraction",
    )
    teacher_public_heuristic_loss_values = _numeric_values(records_for_learning, "teacher_public_heuristic_loss")
    teacher_public_heuristic_supported_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_supported_fraction",
    )
    teacher_public_heuristic_top1_mass_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_top1_mass",
    )
    teacher_public_heuristic_target_entropy_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_target_entropy",
    )
    policy_anchor_coef_active_values = _numeric_values(records_for_learning, "policy_anchor_coef_active")
    policy_anchor_top_action_coef_active_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_coef_active",
    )
    policy_anchor_loss_values = _numeric_values(records_for_learning, "policy_anchor_loss")
    policy_anchor_weighted_loss_values = _numeric_values(records_for_learning, "policy_anchor_weighted_loss")
    policy_anchor_kl_mean_values = _numeric_values(records_for_learning, "policy_anchor_kl_mean")
    policy_anchor_kl_p95_values = _numeric_values(records_for_learning, "policy_anchor_kl_p95")
    policy_anchor_top_action_loss_values = _numeric_values(records_for_learning, "policy_anchor_top_action_loss")
    policy_anchor_top_action_loss_p95_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_loss_p95",
    )
    policy_anchor_top_action_agreement_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_agreement",
    )
    main_move_fraction_values = _fraction_values(scalars, "collector_main_move_actions", "collector_total_actions")
    teacher_tactical_row_fraction_values = _fraction_values(
        scalars,
        "collector_teacher_tactical_row_count",
        "collector_total_actions",
    )
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

    warnings: list[str] = []
    if actor_heuristic_values and max(actor_heuristic_values) > 0.0:
        warnings.append("actor_heuristic_fraction_active was nonzero; focal actions were not pure model-policy rows")
    if heuristic_mix_values and max(heuristic_mix_values) > 0.0:
        warnings.append("heuristic_public_mix_fraction_active was nonzero; eval/train pressure includes B2 heuristic")
    if actor_lag_warning_values and max(actor_lag_warning_values) > _LEARNER_ACTOR_LAG_WARN_THRESHOLD:
        warnings.append(
            f"{actor_lag_warning_source} exceeded {_LEARNER_ACTOR_LAG_WARN_THRESHOLD:g}; actor policy may be stale"
        )

    latest_champion_pool_size = None if not pfsp_champion_pool_size_values else pfsp_champion_pool_size_values[-1]
    latest_snapshot_env_fraction = (
        None if not pfsp_snapshot_env_fraction_values else pfsp_snapshot_env_fraction_values[-1]
    )
    latest_recent_env_fraction = None if not pfsp_recent_env_fraction_values else pfsp_recent_env_fraction_values[-1]
    if promotion_gate["attempt_count"] > 0 and promotion_gate["passed_count"] == 0:
        if latest_champion_pool_size is not None and latest_champion_pool_size > 0.0:
            warnings.append(
                "promotion gate never passed; champion pool is populated by imported/bootstrap champions, "
                "not promoted trained champions"
            )
        elif (
            latest_champion_pool_size == 0.0
            and latest_snapshot_env_fraction is not None
            and latest_snapshot_env_fraction > 0.0
        ):
            warnings.append(
                "promotion gate never passed; no trained champions were admitted, but probationary snapshot "
                "sampling was active"
            )
        else:
            warnings.append("promotion gate never passed; league did not admit any trained champions")
    if int(promotion_gate["consecutive_failure_count"]) >= 3:
        warnings.append(
            "promotion gate failed "
            f"{int(promotion_gate['consecutive_failure_count'])} consecutive attempts through latest update"
        )
    if vtrace_rho_values and max(vtrace_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; behavior/evaluation log-probs may be mismatched"
        )
    train_rho_tail_values = vtrace_train_rho_p99_values or vtrace_train_rho_p95_values
    max_train_rho_tail = None if not train_rho_tail_values else max(train_rho_tail_values)
    if vtrace_rho_p99_values and max(vtrace_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
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
    if vtrace_train_rho_values and max(vtrace_train_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; train-mask behavior/evaluation log-probs may be mismatched"
        )
    if vtrace_train_rho_p95_values and max(vtrace_train_rho_p95_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_p95 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if vtrace_train_rho_p99_values and max(vtrace_train_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_p99 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if vtrace_clip_rate_values and max(vtrace_clip_rate_values) > _VTRACE_CLIP_RATE_WARN_THRESHOLD:
        warnings.append(
            f"vtrace_clip_rate exceeded {_VTRACE_CLIP_RATE_WARN_THRESHOLD:g}; policy updates are heavily clipped"
        )
    if (
        train_logp_delta_abs_p99_values
        and max(train_logp_delta_abs_p99_values) > _TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD
    ):
        warnings.append(
            "target_behavior_train_logp_delta_abs_p99 exceeded "
            f"{_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD:g}; learner and behavior log-probs diverged on train rows"
        )
    if (
        max_consecutive_main_move_values
        and max(max_consecutive_main_move_values) > _MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD
    ):
        warnings.append(
            "collector_max_consecutive_main_moves exceeded "
            f"{_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD:g}; repeated main-move transitions or counter drift suspected"
        )
    chosen_pass_train_fraction_last = _last_window_mean(chosen_pass_train_fraction_values, window=20)
    if chosen_pass_train_fraction_last is not None and chosen_pass_train_fraction_last > 0.5:
        warnings.append("chosen_pass_train_fraction averaged above 0.5 in the latest window; pass-collapse suspected")
    chosen_mulligan_select_share_last = _last_window_mean(chosen_mulligan_select_share_values, window=20)
    if (
        chosen_mulligan_select_share_last is not None
        and chosen_mulligan_select_share_last > _MULLIGAN_SELECT_SHARE_WARN_THRESHOLD
    ):
        warnings.append(
            "chosen_mulligan_select share among mulligan actions is high in the latest window; "
            "mulligan-confirm collapse suspected"
        )
    pass_with_nonpass_total_fraction_last = _last_window_mean(pass_with_nonpass_total_fraction_values, window=20)
    if pass_with_nonpass_total_fraction_last is not None and pass_with_nonpass_total_fraction_last > 0.35:
        warnings.append(
            "collector pass-with-nonpass fraction is high in the latest window; policy may be avoiding play"
        )
    teacher_public_heuristic_coef_active_last = _last_window_mean(
        teacher_public_heuristic_coef_active_values, window=20
    )
    teacher_public_heuristic_supported_last = _last_window_mean(teacher_public_heuristic_supported_values, window=20)
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
    teacher_hand_coef_active_last = _last_window_mean(teacher_hand_coef_active_values, window=20)
    teacher_hand_supported_last = _last_window_mean(teacher_hand_supported_values, window=20)
    if (
        teacher_hand_coef_active_last is not None
        and teacher_hand_coef_active_last > 0.0
        and (teacher_hand_supported_last is None or teacher_hand_supported_last < _TEACHER_SUPPORTED_WARN_THRESHOLD)
    ):
        warnings.append(
            "teacher_hand_coef_active was nonzero but hand-target support was near zero; "
            "hand metadata or factorized same-family arg0 references may be missing"
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
        "teacher_guidance": {
            "teacher_public_heuristic_coef_active": _window_summary(
                teacher_public_heuristic_coef_active_values,
                window=20,
            ),
            "teacher_hand_coef_active": _window_summary(
                teacher_hand_coef_active_values,
                window=20,
            ),
            "teacher_aux_loss": _window_summary(teacher_aux_loss_values, window=20),
            "teacher_main_play_character_slot_accuracy": _window_summary(
                teacher_main_play_slot_accuracy_values,
                window=20,
            ),
            "teacher_hand_accuracy": _window_summary(teacher_hand_accuracy_values, window=20),
            "teacher_main_play_character_hand_accuracy": _window_summary(
                teacher_main_play_hand_accuracy_values,
                window=20,
            ),
            "teacher_clock_from_hand_accuracy": _window_summary(
                teacher_clock_hand_accuracy_values,
                window=20,
            ),
            "teacher_hand_loss": _window_summary(teacher_hand_loss_values, window=20),
            "teacher_hand_supported_fraction": _window_summary(teacher_hand_supported_values, window=20),
            "teacher_same_family_action_accuracy": _window_summary(
                teacher_same_family_action_accuracy_values,
                window=20,
            ),
            "teacher_same_family_main_play_character_accuracy": _window_summary(
                teacher_same_family_main_play_accuracy_values,
                window=20,
            ),
            "teacher_action_margin_mean": _window_summary(
                teacher_action_margin_mean_values,
                window=20,
            ),
            "teacher_action_margin_satisfied_fraction": _window_summary(
                teacher_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_same_family_action_margin_mean": _window_summary(
                teacher_same_family_action_margin_mean_values,
                window=20,
            ),
            "teacher_same_family_action_margin_satisfied_fraction": _window_summary(
                teacher_same_family_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_public_heuristic_loss": _window_summary(
                teacher_public_heuristic_loss_values,
                window=20,
            ),
            "teacher_public_heuristic_supported_fraction": _window_summary(
                teacher_public_heuristic_supported_values,
                window=20,
            ),
            "teacher_public_heuristic_top1_mass": _window_summary(
                teacher_public_heuristic_top1_mass_values,
                window=20,
            ),
            "teacher_public_heuristic_target_entropy": _window_summary(
                teacher_public_heuristic_target_entropy_values,
                window=20,
            ),
            "teacher_tactical_row_fraction_of_total": _window_summary(
                teacher_tactical_row_fraction_values,
                window=20,
            ),
            "policy_anchor_coef_active": _window_summary(policy_anchor_coef_active_values, window=20),
            "policy_anchor_top_action_coef_active": _window_summary(
                policy_anchor_top_action_coef_active_values,
                window=20,
            ),
            "policy_anchor_loss": _window_summary(policy_anchor_loss_values, window=20),
            "policy_anchor_weighted_loss": _window_summary(policy_anchor_weighted_loss_values, window=20),
            "policy_anchor_kl_mean": _window_summary(policy_anchor_kl_mean_values, window=20),
            "policy_anchor_kl_p95": _window_summary(policy_anchor_kl_p95_values, window=20),
            "policy_anchor_top_action_loss": _window_summary(policy_anchor_top_action_loss_values, window=20),
            "policy_anchor_top_action_loss_p95": _window_summary(
                policy_anchor_top_action_loss_p95_values,
                window=20,
            ),
            "policy_anchor_top_action_agreement": _window_summary(
                policy_anchor_top_action_agreement_values,
                window=20,
            ),
            "max_teacher_public_heuristic_coef_active": None
            if not teacher_public_heuristic_coef_active_values
            else max(teacher_public_heuristic_coef_active_values),
            "max_teacher_hand_coef_active": None
            if not teacher_hand_coef_active_values
            else max(teacher_hand_coef_active_values),
            "max_teacher_public_heuristic_supported_fraction": None
            if not teacher_public_heuristic_supported_values
            else max(teacher_public_heuristic_supported_values),
            "max_teacher_hand_supported_fraction": None
            if not teacher_hand_supported_values
            else max(teacher_hand_supported_values),
        },
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
        "actor_model_sync": {
            "policy_version_lag_p50": _window_summary(policy_version_lag_p50_values, window=20),
            "policy_version_lag_p90": _window_summary(policy_version_lag_p90_values, window=20),
            "max_policy_version_lag_p90": None
            if not policy_version_lag_p90_values
            else max(policy_version_lag_p90_values),
            "learner_actor_update_lag_p50": _window_summary(learner_actor_update_lag_p50_values, window=20),
            "learner_actor_update_lag_p90": _window_summary(learner_actor_update_lag_p90_values, window=20),
            "max_learner_actor_update_lag_p90": None
            if not learner_actor_update_lag_p90_values
            else max(learner_actor_update_lag_p90_values),
            "lag_warning_source": actor_lag_warning_source,
            "learner_to_actor_update_lag": _window_summary(actor_lag_warning_values, window=20),
            "max_learner_to_actor_update_lag": None if not actor_lag_warning_values else max(actor_lag_warning_values),
        },
        "league_sync": {
            "league_update_lag": _window_summary(league_update_lag_values, window=20),
            "max_league_update_lag": None if not league_update_lag_values else max(league_update_lag_values),
        },
        "off_policy": {
            "vtrace_rho_mean": _window_summary(vtrace_rho_values, window=20),
            "vtrace_rho_p99": _window_summary(vtrace_rho_p99_values, window=20),
            "vtrace_train_rho_mean": _window_summary(vtrace_train_rho_values, window=20),
            "vtrace_train_rho_p95": _window_summary(vtrace_train_rho_p95_values, window=20),
            "vtrace_train_rho_p99": _window_summary(vtrace_train_rho_p99_values, window=20),
            "vtrace_clip_rate": _window_summary(vtrace_clip_rate_values, window=20),
            "target_behavior_logp_delta_abs_mean": _window_summary(logp_delta_abs_values, window=20),
            "target_behavior_logp_delta_abs_p99": _window_summary(logp_delta_abs_p99_values, window=20),
            "target_behavior_train_logp_delta_abs_mean": _window_summary(train_logp_delta_abs_values, window=20),
            "target_behavior_train_logp_delta_abs_p99": _window_summary(train_logp_delta_abs_p99_values, window=20),
            "max_vtrace_rho_mean": None if not vtrace_rho_values else max(vtrace_rho_values),
            "max_vtrace_rho_p99": None if not vtrace_rho_p99_values else max(vtrace_rho_p99_values),
            "max_vtrace_train_rho_mean": None if not vtrace_train_rho_values else max(vtrace_train_rho_values),
            "max_vtrace_train_rho_p95": None if not vtrace_train_rho_p95_values else max(vtrace_train_rho_p95_values),
            "max_vtrace_train_rho_p99": None if not vtrace_train_rho_p99_values else max(vtrace_train_rho_p99_values),
            "max_vtrace_clip_rate": None if not vtrace_clip_rate_values else max(vtrace_clip_rate_values),
            "max_target_behavior_logp_delta_abs_mean": None
            if not logp_delta_abs_values
            else max(logp_delta_abs_values),
            "max_target_behavior_logp_delta_abs_p99": None
            if not logp_delta_abs_p99_values
            else max(logp_delta_abs_p99_values),
            "max_target_behavior_train_logp_delta_abs_mean": None
            if not train_logp_delta_abs_values
            else max(train_logp_delta_abs_values),
            "max_target_behavior_train_logp_delta_abs_p99": None
            if not train_logp_delta_abs_p99_values
            else max(train_logp_delta_abs_p99_values),
            "stale_policy_lag_source": actor_lag_warning_source,
            "stale_policy_lag_correlations": {
                key: {
                    "paired_update_count": len(pairs),
                    "pearson": _pearson_correlation(pairs),
                }
                for key, pairs in stale_policy_pairs.items()
            },
        },
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
