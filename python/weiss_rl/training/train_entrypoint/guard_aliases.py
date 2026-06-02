"""Checkpoint-guard legacy aliases for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Mapping

CHECKPOINT_GUARD_ALIASES: Mapping[str, str] = {
    "_checkpoint_candidate_metric": "checkpoint_candidate_metric",
    "_confirmatory_dev_eval_request": "confirmatory_dev_eval_request",
    "_confirmatory_dev_eval_target_pairs": "confirmatory_dev_eval_target_pairs",
    "_dev_eval_aggregate_score": "dev_eval_aggregate_score",
    "_dev_eval_confidence_stats": "dev_eval_confidence_stats",
    "_dev_eval_ineligibility_reasons": "dev_eval_ineligibility_reasons",
    "_dev_eval_metric_eligible": "dev_eval_metric_eligible",
    "_dev_eval_worst_natural_timeout_rate": "dev_eval_worst_natural_timeout_rate",
    "_dev_eval_worst_no_progress_timeout_rate": "dev_eval_worst_no_progress_timeout_rate",
    "_dev_eval_worst_reason_rate": "dev_eval_worst_reason_rate",
    "_dev_eval_worst_stall_rate": "dev_eval_worst_stall_rate",
    "_dev_eval_worst_truncation_rate": "dev_eval_worst_truncation_rate",
    "_expand_periodic_dev_eval_paired_seeds": "expand_periodic_dev_eval_paired_seeds",
    "_should_promote_best_checkpoint": "should_promote_best_checkpoint",
    "_summary_rate": "summary_rate",
}
