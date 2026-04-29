"""Dev-eval scoring and checkpoint decision helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from weiss_rl.config import StackConfig

_CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL = 0.1
_CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS = 0.05
B2_FLATLINE_WINDOW = 3
B2_FLATLINE_MAX_DELTA = 0.02
B2_FLATLINE_LOW_SCORE = 0.35
B2_ACTION_WARNING_SCORE_THRESHOLD = 0.25
B2_ACTION_WARNING_MAIN_MOVE_RATE = 0.45
B2_ACTION_WARNING_PASS_NONPASS_RATE = 0.05


def dev_eval_aggregate_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    aggregate_score = dev_eval_summary.get("aggregate_score")
    if isinstance(aggregate_score, (int, float)) and np.isfinite(float(aggregate_score)):
        return float(aggregate_score)
    uncertainty = dev_eval_summary.get("uncertainty")
    if isinstance(uncertainty, Mapping):
        mean_value = uncertainty.get("mean")
        if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
            return float(mean_value)
    return None


def weighted_dev_eval_aggregate(
    anchor_scores: Mapping[str, float],
    *,
    anchor_weights: Mapping[str, float],
) -> tuple[float, dict[str, float], float]:
    if not anchor_scores:
        return 0.0, {}, 0.0
    active_weights: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for anchor_name, score in anchor_scores.items():
        weight = float(anchor_weights.get(anchor_name, 1.0))
        if weight <= 0.0:
            continue
        active_weights[str(anchor_name)] = weight
        weighted_sum += float(score) * weight
        total_weight += weight
    if total_weight <= 0.0:
        for anchor_name, score in anchor_scores.items():
            active_weights[str(anchor_name)] = 1.0
            weighted_sum += float(score)
            total_weight += 1.0
    return float(weighted_sum / total_weight), active_weights, float(total_weight)


def dev_eval_surface(dev_eval_summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if dev_eval_summary is None:
        return {}
    surface = dev_eval_summary.get("evaluation_surface")
    if isinstance(surface, Mapping):
        return cast(Mapping[str, Any], surface)
    return {}


def dev_eval_is_authoritative(dev_eval_summary: Mapping[str, Any] | None) -> bool:
    if dev_eval_summary is None:
        return False
    surface = dev_eval_surface(dev_eval_summary)
    authoritative = surface.get("authoritative")
    if isinstance(authoritative, bool):
        return authoritative
    # Older summaries predate the surface contract and are canonical scalar by construction.
    return True


def extract_anchor_payload(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> Mapping[str, Any] | None:
    if dev_eval_summary is None:
        return None
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    anchor_payload = anchors.get(anchor_name)
    if not isinstance(anchor_payload, Mapping):
        return None
    return cast(Mapping[str, Any], anchor_payload)


def extract_anchor_score(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if isinstance(anchor_scores, Mapping):
        score = anchor_scores.get(anchor_name)
        if isinstance(score, (int, float)) and np.isfinite(float(score)):
            return float(score)
    anchor_payload = extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    uncertainty = anchor_payload.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        return None
    mean_value = uncertainty.get("mean")
    if isinstance(mean_value, (int, float)) and np.isfinite(float(mean_value)):
        return float(mean_value)
    return None


def extract_anchor_summary(dev_eval_summary: Mapping[str, Any] | None, anchor_name: str) -> Mapping[str, Any] | None:
    anchor_payload = extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    summary = anchor_payload.get("summary")
    if not isinstance(summary, Mapping):
        return None
    return cast(Mapping[str, Any], summary)


def extract_anchor_uncertainty(
    dev_eval_summary: Mapping[str, Any] | None, anchor_name: str
) -> Mapping[str, Any] | None:
    anchor_payload = extract_anchor_payload(dev_eval_summary, anchor_name)
    if anchor_payload is None:
        return None
    uncertainty = anchor_payload.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        return None
    return cast(Mapping[str, Any], uncertainty)


def summary_fraction(
    summary: Mapping[str, Any] | None,
    *,
    numerator_key: str,
    denominator_key: str,
) -> float | None:
    if summary is None:
        return None
    numerator = summary.get(numerator_key)
    denominator = summary.get(denominator_key)
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def b2_recent_scores_from_persisted_summaries(
    summaries: Mapping[str, Any],
    *,
    current_policy_id: str,
    heuristic_public_policy_id: str,
) -> list[float]:
    scored_entries: list[tuple[int, float]] = []
    for policy_id, raw_entry in summaries.items():
        if policy_id == current_policy_id or not isinstance(raw_entry, Mapping):
            continue
        update_count = raw_entry.get("update_count")
        b2_payload = raw_entry.get("b2")
        score: float | None = None
        if isinstance(b2_payload, Mapping):
            raw_score = b2_payload.get("score")
            if isinstance(raw_score, (int, float)) and np.isfinite(float(raw_score)):
                score = float(raw_score)
        if score is None:
            raw_anchor_scores = raw_entry.get("anchor_scores")
            if isinstance(raw_anchor_scores, Mapping):
                raw_score = raw_anchor_scores.get(heuristic_public_policy_id)
                if isinstance(raw_score, (int, float)) and np.isfinite(float(raw_score)):
                    score = float(raw_score)
        if score is None:
            continue
        scored_entries.append((0 if not isinstance(update_count, int) else int(update_count), score))
    scored_entries.sort(key=lambda item: item[0])
    return [score for _update_count, score in scored_entries[-(B2_FLATLINE_WINDOW - 1) :]]


def build_b2_warning_flags(
    *,
    current_score: float | None,
    current_summary: Mapping[str, Any] | None,
    recent_scores: Sequence[float],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if current_score is None:
        return warnings

    score_window = [*recent_scores, float(current_score)]
    if (
        len(score_window) >= B2_FLATLINE_WINDOW
        and max(score_window) - min(score_window) <= B2_FLATLINE_MAX_DELTA
        and float(current_score) <= B2_FLATLINE_LOW_SCORE
    ):
        warnings.append(
            {
                "kind": "b2_flatline_v1",
                "recent_eval_count": len(score_window),
                "min_score": float(min(score_window)),
                "max_score": float(max(score_window)),
                "score_delta": float(max(score_window) - min(score_window)),
            }
        )

    main_move_rate = summary_fraction(
        current_summary, numerator_key="main_move_actions", denominator_key="total_actions"
    )
    pass_nonpass_rate = summary_fraction(
        current_summary,
        numerator_key="pass_with_nonpass_available",
        denominator_key="total_actions",
    )
    max_consecutive_main_moves = None if current_summary is None else current_summary.get("max_consecutive_main_moves")
    if float(current_score) <= B2_ACTION_WARNING_SCORE_THRESHOLD and (
        (main_move_rate is not None and main_move_rate >= B2_ACTION_WARNING_MAIN_MOVE_RATE)
        or (pass_nonpass_rate is not None and pass_nonpass_rate >= B2_ACTION_WARNING_PASS_NONPASS_RATE)
    ):
        warning_payload: dict[str, Any] = {
            "kind": "b2_action_family_warning_v1",
            "score": float(current_score),
        }
        if main_move_rate is not None:
            warning_payload["main_move_rate"] = float(main_move_rate)
        if pass_nonpass_rate is not None:
            warning_payload["pass_with_nonpass_available_rate"] = float(pass_nonpass_rate)
        if isinstance(max_consecutive_main_moves, (int, float)):
            warning_payload["max_consecutive_main_moves"] = int(max_consecutive_main_moves)
        warnings.append(warning_payload)
    return warnings


def dev_eval_worst_truncation_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get("worst_truncation_rate")
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        games = summary.get("games")
        truncations = summary.get("truncations")
        if not isinstance(games, (int, float)) or not isinstance(truncations, (int, float)):
            continue
        if float(games) <= 0:
            continue
        rate = float(truncations) / float(games)
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def dev_eval_worst_reason_rate(
    dev_eval_summary: Mapping[str, Any] | None,
    *,
    summary_key: str,
    stall_monitor_key: str,
) -> float | None:
    if dev_eval_summary is None:
        return None
    stall_monitor = dev_eval_summary.get("stall_monitor")
    if isinstance(stall_monitor, Mapping):
        worst_rate = stall_monitor.get(stall_monitor_key)
        if isinstance(worst_rate, (int, float)) and np.isfinite(float(worst_rate)):
            return float(worst_rate)
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return None
    worst_rate: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        summary = anchor_payload.get("summary")
        if not isinstance(summary, Mapping):
            continue
        rate = summary_rate(summary, summary_key)
        if rate is None:
            continue
        worst_rate = rate if worst_rate is None else max(worst_rate, rate)
    return worst_rate


def dev_eval_worst_no_progress_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="no_progress_timeouts",
        stall_monitor_key="worst_no_progress_timeout_rate",
    )


def dev_eval_worst_natural_timeout_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    return dev_eval_worst_reason_rate(
        dev_eval_summary,
        summary_key="natural_timeouts",
        stall_monitor_key="worst_natural_timeout_rate",
    )


def dev_eval_worst_stall_rate(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    no_progress_rate = dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    if no_progress_rate is not None:
        return no_progress_rate
    return dev_eval_worst_truncation_rate(dev_eval_summary)


def dev_eval_confidence_stats(dev_eval_summary: Mapping[str, Any] | None) -> dict[str, float | None]:
    stats = {
        "min_prob_gt_half": None,
        "max_prob_lt_half": None,
        "max_ci_half_width": None,
    }
    if dev_eval_summary is None:
        return stats
    anchors = dev_eval_summary.get("anchors")
    if not isinstance(anchors, Mapping):
        return stats
    min_prob_gt_half: float | None = None
    max_prob_lt_half: float | None = None
    max_ci_half_width: float | None = None
    for anchor_payload in anchors.values():
        if not isinstance(anchor_payload, Mapping):
            continue
        uncertainty = anchor_payload.get("uncertainty")
        if not isinstance(uncertainty, Mapping):
            continue
        prob_gt_half = uncertainty.get("prob_gt_half")
        prob_lt_half = uncertainty.get("prob_lt_half")
        ci_half_width = uncertainty.get("ci_half_width")
        if isinstance(prob_gt_half, (int, float)) and np.isfinite(float(prob_gt_half)):
            min_prob_gt_half = (
                float(prob_gt_half) if min_prob_gt_half is None else min(min_prob_gt_half, float(prob_gt_half))
            )
        if isinstance(prob_lt_half, (int, float)) and np.isfinite(float(prob_lt_half)):
            max_prob_lt_half = (
                float(prob_lt_half) if max_prob_lt_half is None else max(max_prob_lt_half, float(prob_lt_half))
            )
        if isinstance(ci_half_width, (int, float)) and np.isfinite(float(ci_half_width)):
            max_ci_half_width = (
                float(ci_half_width) if max_ci_half_width is None else max(max_ci_half_width, float(ci_half_width))
            )
    stats["min_prob_gt_half"] = min_prob_gt_half
    stats["max_prob_lt_half"] = max_prob_lt_half
    stats["max_ci_half_width"] = max_ci_half_width
    return stats


def dev_eval_ineligibility_reasons(
    stack: StackConfig,
    *,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if dev_eval_summary is None:
        return ("missing",)
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return ("missing_score",)
    if not dev_eval_is_authoritative(dev_eval_summary):
        return ("non_authoritative",)
    curriculum = stack.config.curriculum
    if curriculum is None:
        return ()
    reasons: list[str] = []
    if curriculum.stall_monitor.enabled:
        worst_rate = dev_eval_worst_stall_rate(dev_eval_summary)
        if worst_rate is not None and worst_rate >= float(curriculum.stall_monitor.truncation_rate_threshold):
            reasons.append("truncation")
    checkpoint_guard = curriculum.checkpoint_guard
    if checkpoint_guard.enabled:
        confidence = dev_eval_confidence_stats(dev_eval_summary)
        min_prob_gt_half = confidence["min_prob_gt_half"]
        max_ci_half_width = confidence["max_ci_half_width"]
        if min_prob_gt_half is not None and (
            float(min_prob_gt_half) < float(checkpoint_guard.promote_min_prob_gt_half)
        ):
            max_prob_lt_half = confidence["max_prob_lt_half"]
            tolerated_prob_lt_half = max(0.0, 1.0 - float(checkpoint_guard.promote_min_prob_gt_half))
            if (
                float(min_prob_gt_half) > 0.0
                or max_prob_lt_half is None
                or float(max_prob_lt_half) > tolerated_prob_lt_half
            ):
                reasons.append("confidence_prob")
        if max_ci_half_width is not None and (
            float(max_ci_half_width) > float(checkpoint_guard.promote_max_ci_half_width)
        ):
            reasons.append("confidence_ci")
    return tuple(reasons)


def dev_eval_metric_eligible(stack: StackConfig, *, dev_eval_summary: Mapping[str, Any] | None) -> bool:
    return not dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)


def confirmatory_dev_eval_target_pairs(stack: StackConfig) -> int:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("training stack is missing evaluation config")
    base_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    max_pairs = int(evaluation.final_matrix_stage2_adaptive_max_paired_seeds)
    return max(base_pairs, min(max_pairs, max(32, base_pairs * 4)))


def confirmatory_dev_eval_request(
    *,
    stack: StackConfig,
    existing_best_record: Mapping[str, Any] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    reasons = dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    if any(reason not in {"confidence_prob", "confidence_ci"} for reason in reasons):
        return None
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if float(current_score) < float(checkpoint_guard.min_best_score):
        return None

    existing_metric_kind = ""
    existing_metric_value: float | None = None
    score_shortfall = 0.0
    if existing_best_record is not None:
        existing_metric_kind = str(existing_best_record.get("metric_kind", "")).strip()
        raw_existing_metric_value = existing_best_record.get("metric_value")
        if isinstance(raw_existing_metric_value, (int, float)) and np.isfinite(float(raw_existing_metric_value)):
            existing_metric_value = float(raw_existing_metric_value)
            score_shortfall = max(0.0, existing_metric_value - float(current_score))
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall > 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        return None

    confidence = dev_eval_confidence_stats(dev_eval_summary)
    confirmatory_reasons: list[str] = []
    prob_shortfall = 0.0
    if "confidence_prob" in reasons:
        min_prob_gt_half = confidence["min_prob_gt_half"]
        if min_prob_gt_half is None:
            return None
        prob_shortfall = max(0.0, float(checkpoint_guard.promote_min_prob_gt_half) - float(min_prob_gt_half))
        if prob_shortfall <= _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
            confirmatory_reasons.append("confidence_prob")
    ci_excess = 0.0
    if "confidence_ci" in reasons:
        max_ci_half_width = confidence["max_ci_half_width"]
        if max_ci_half_width is None:
            return None
        ci_excess = max(0.0, float(max_ci_half_width) - float(checkpoint_guard.promote_max_ci_half_width))
        if ci_excess <= _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
            confirmatory_reasons.append("confidence_ci")
    if (
        existing_metric_kind == "dev_eval_mean"
        and existing_metric_value is not None
        and score_shortfall > 0.0
        and score_shortfall <= 2.0 * float(checkpoint_guard.rollback_score_margin)
    ):
        confirmatory_reasons.append("score_drop")
    if not confirmatory_reasons:
        return None
    if prob_shortfall > _CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL:
        return None
    if ci_excess > _CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
        return None

    return {
        "reasons": confirmatory_reasons,
        "current_score": float(current_score),
        "existing_best_score": existing_metric_value,
        "prob_shortfall": prob_shortfall,
        "ci_excess": ci_excess,
        "target_pairs": confirmatory_dev_eval_target_pairs(stack),
    }


def checkpoint_candidate_metric(
    *,
    stack: StackConfig,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str | None, float | None]:
    if dev_eval_metric_eligible(stack, dev_eval_summary=dev_eval_summary):
        aggregate_score = dev_eval_aggregate_score(dev_eval_summary)
        if aggregate_score is not None:
            return "dev_eval_mean", aggregate_score
    evaluation = stack.config.evaluation
    if evaluation is not None and int(evaluation.periodic_dev_eval_interval_updates) > 0:
        return None, None
    if latest_metrics is not None:
        loss_value = latest_metrics.get("loss")
        if isinstance(loss_value, (int, float)) and np.isfinite(float(loss_value)):
            return "training_loss", float(loss_value)
    return None, None


def should_promote_best_checkpoint(
    *,
    existing_record: Mapping[str, Any] | None,
    candidate_kind: str | None,
    candidate_value: float | None,
) -> bool:
    if candidate_kind is None:
        return False
    if existing_record is None:
        return True
    existing_kind = existing_record.get("metric_kind")
    existing_value = existing_record.get("metric_value")
    if candidate_kind == "dev_eval_mean":
        if existing_kind != "dev_eval_mean":
            return True
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) > float(existing_value)
    if candidate_kind == "training_loss":
        if existing_kind == "dev_eval_mean":
            return False
        if not isinstance(existing_value, (int, float)):
            return True
        return float(candidate_value) < float(existing_value)
    return False


def should_update_secondary_b2_record(
    *,
    existing_record: Mapping[str, Any] | None,
    candidate_b2_score: float,
    candidate_aggregate_score: float | None,
    update_count: int,
    policy_version: int,
) -> bool:
    if existing_record is None:
        return True
    existing_metric = existing_record.get("metric_value")
    if not isinstance(existing_metric, (int, float)) or not np.isfinite(float(existing_metric)):
        return True
    existing_b2_score = float(existing_metric)
    if float(candidate_b2_score) > existing_b2_score:
        return True
    if float(candidate_b2_score) < existing_b2_score:
        return False

    existing_aggregate = existing_record.get("aggregate_score")
    if (
        candidate_aggregate_score is not None
        and isinstance(existing_aggregate, (int, float))
        and np.isfinite(float(existing_aggregate))
    ):
        if float(candidate_aggregate_score) > float(existing_aggregate):
            return True
        if float(candidate_aggregate_score) < float(existing_aggregate):
            return False

    existing_update = existing_record.get("update_count")
    if isinstance(existing_update, int) and int(update_count) != int(existing_update):
        return int(update_count) > int(existing_update)
    existing_version = existing_record.get("policy_version")
    if isinstance(existing_version, int) and int(policy_version) != int(existing_version):
        return int(policy_version) > int(existing_version)
    return False
