from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.training.checkpointing.guards.dev_eval_metrics import (
    DevEvalConfidenceStats,
    DevEvalTimeoutRates,
    collect_dev_eval_confidence_stats,
    collect_dev_eval_timeout_rates,
    dev_eval_aggregate_score,
    dev_eval_confidence_stats,
    dev_eval_worst_anchor_mean,
    dev_eval_worst_natural_timeout_rate,
    dev_eval_worst_no_progress_timeout_rate,
    dev_eval_worst_reason_rate,
    dev_eval_worst_stall_rate,
    dev_eval_worst_truncation_rate,
    summary_rate,
)

_U64_MASK = (1 << 64) - 1
CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL = 0.1
CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS = 0.05
CONFIRMATORY_DEV_EVAL_MIN_WORST_ANCHOR_MEAN = 0.45


@dataclass(frozen=True, slots=True)
class DevEvalEligibilityAssessment:
    score: float | None
    timeout_rates: DevEvalTimeoutRates
    confidence: DevEvalConfidenceStats
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.reasons


def dev_eval_ineligibility_reasons(
    stack: Any,
    *,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    return assess_dev_eval_metric_eligibility(stack, dev_eval_summary=dev_eval_summary).reasons


def assess_dev_eval_metric_eligibility(
    stack: Any,
    *,
    dev_eval_summary: Mapping[str, Any] | None,
) -> DevEvalEligibilityAssessment:
    timeout_rates = collect_dev_eval_timeout_rates(dev_eval_summary)
    confidence = collect_dev_eval_confidence_stats(dev_eval_summary)
    if dev_eval_summary is None:
        return DevEvalEligibilityAssessment(
            score=None,
            timeout_rates=timeout_rates,
            confidence=confidence,
            reasons=("missing",),
        )
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return DevEvalEligibilityAssessment(
            score=None,
            timeout_rates=timeout_rates,
            confidence=confidence,
            reasons=("missing_score",),
        )
    curriculum = stack.config.curriculum
    if curriculum is None:
        return DevEvalEligibilityAssessment(
            score=float(current_score),
            timeout_rates=timeout_rates,
            confidence=confidence,
            reasons=(),
        )
    reasons: list[str] = []
    if bool(curriculum.stall_monitor.enabled):
        worst_rate = timeout_rates.worst_stall_rate
        if worst_rate is not None and worst_rate >= float(curriculum.stall_monitor.truncation_rate_threshold):
            reasons.append("truncation")
    checkpoint_guard = curriculum.checkpoint_guard
    if bool(checkpoint_guard.enabled):
        min_prob_gt_half = confidence.min_prob_gt_half
        max_ci_half_width = confidence.max_ci_half_width
        if min_prob_gt_half is not None and (
            float(min_prob_gt_half) < float(checkpoint_guard.promote_min_prob_gt_half)
        ):
            reasons.append("confidence_prob")
        if max_ci_half_width is not None and (
            float(max_ci_half_width) > float(checkpoint_guard.promote_max_ci_half_width)
        ):
            reasons.append("confidence_ci")
    return DevEvalEligibilityAssessment(
        score=float(current_score),
        timeout_rates=timeout_rates,
        confidence=confidence,
        reasons=tuple(reasons),
    )


def dev_eval_metric_eligible(stack: Any, *, dev_eval_summary: Mapping[str, Any] | None) -> bool:
    return assess_dev_eval_metric_eligibility(stack, dev_eval_summary=dev_eval_summary).eligible


def confirmatory_dev_eval_target_pairs(stack: Any) -> int:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    base_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    max_pairs = int(evaluation.final_matrix_stage2_adaptive_max_paired_seeds)
    return max(base_pairs, min(max_pairs, max(32, base_pairs * 4)))


def expand_periodic_dev_eval_paired_seeds(
    base_paired_seeds: Sequence[int],
    *,
    requested_pairs: int,
    seed_file_sha256: str,
    update_count: int,
    policy_version: int,
    scope: str,
) -> list[int]:
    requested_pairs_i = int(requested_pairs)
    paired_seeds = [int(seed) for seed in base_paired_seeds[:requested_pairs_i]]
    seen = set(paired_seeds)
    extra_index = 0
    while len(paired_seeds) < requested_pairs_i:
        derived_seed = (
            stable_hash64(
                canonical_json_bytes(
                    {
                        "kind": "periodic_dev_eval_confirmatory_seed_v1",
                        "scope": str(scope),
                        "seed_file_sha256": str(seed_file_sha256),
                        "update_count": int(update_count),
                        "policy_version": int(policy_version),
                        "extra_index": int(extra_index),
                    }
                )
            )
            & _U64_MASK
        )
        extra_index += 1
        if derived_seed in seen:
            continue
        paired_seeds.append(int(derived_seed))
        seen.add(int(derived_seed))
    return paired_seeds


def confirmatory_dev_eval_request(
    *,
    stack: Any,
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
    worst_anchor_mean = dev_eval_worst_anchor_mean(dev_eval_summary)
    confirmatory_reasons: list[str] = []
    prob_shortfall = 0.0
    confidence_prob_confirmable = False
    if "confidence_prob" in reasons:
        min_prob_gt_half = confidence["min_prob_gt_half"]
        if min_prob_gt_half is None:
            return None
        prob_shortfall = max(0.0, float(checkpoint_guard.promote_min_prob_gt_half) - float(min_prob_gt_half))
        confidence_prob_confirmable = prob_shortfall <= CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL or (
            worst_anchor_mean is not None and float(worst_anchor_mean) >= CONFIRMATORY_DEV_EVAL_MIN_WORST_ANCHOR_MEAN
        )
        if confidence_prob_confirmable:
            confirmatory_reasons.append("confidence_prob")
    ci_excess = 0.0
    if "confidence_ci" in reasons:
        max_ci_half_width = confidence["max_ci_half_width"]
        if max_ci_half_width is None:
            return None
        ci_excess = max(0.0, float(max_ci_half_width) - float(checkpoint_guard.promote_max_ci_half_width))
        if ci_excess <= CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
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
    if prob_shortfall > CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL and not confidence_prob_confirmable:
        return None
    if ci_excess > CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS:
        return None

    return {
        "reasons": confirmatory_reasons,
        "current_score": float(current_score),
        "existing_best_score": existing_metric_value,
        "prob_shortfall": prob_shortfall,
        "ci_excess": ci_excess,
        "worst_anchor_mean": worst_anchor_mean,
        "target_pairs": confirmatory_dev_eval_target_pairs(stack),
    }


def checkpoint_candidate_metric(
    *,
    stack: Any,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> tuple[str | None, float | None]:
    assessment = assess_dev_eval_metric_eligibility(stack, dev_eval_summary=dev_eval_summary)
    if assessment.eligible and assessment.score is not None:
        return "dev_eval_mean", assessment.score
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
        return float(cast(float, candidate_value)) > float(existing_value)
    if candidate_kind == "training_loss":
        if existing_kind == "dev_eval_mean":
            return False
        if not isinstance(existing_value, (int, float)):
            return True
        return float(cast(float, candidate_value)) < float(existing_value)
    return False


__all__ = [
    "DevEvalConfidenceStats",
    "DevEvalEligibilityAssessment",
    "DevEvalTimeoutRates",
    "assess_dev_eval_metric_eligibility",
    "checkpoint_candidate_metric",
    "collect_dev_eval_confidence_stats",
    "collect_dev_eval_timeout_rates",
    "confirmatory_dev_eval_request",
    "confirmatory_dev_eval_target_pairs",
    "dev_eval_aggregate_score",
    "dev_eval_confidence_stats",
    "dev_eval_ineligibility_reasons",
    "dev_eval_metric_eligible",
    "dev_eval_worst_anchor_mean",
    "dev_eval_worst_natural_timeout_rate",
    "dev_eval_worst_no_progress_timeout_rate",
    "dev_eval_worst_reason_rate",
    "dev_eval_worst_stall_rate",
    "dev_eval_worst_truncation_rate",
    "expand_periodic_dev_eval_paired_seeds",
    "should_promote_best_checkpoint",
    "summary_rate",
]
