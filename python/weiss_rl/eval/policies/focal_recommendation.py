"""Focal policy recommendation for final eval reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from weiss_rl.eval.policies.dev_eval_summaries import (
    DevEvalSummaryLike,
    canonicalize_dev_eval_summaries,
    normalize_dev_eval_summaries,
)
from weiss_rl.eval.policies.fixed_panel import (
    LEGACY_NO_LEAGUE_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
    heuristic_public_profile_name_for_policy_id,
)
from weiss_rl.eval.policies.registry_view import snapshot_training_policies
from weiss_rl.eval.policies.training_policy_ids import (
    TrainingPolicyId,
    training_policy_sort_key,
    training_policy_tie_break,
    try_parse_training_policy,
)


def recommend_focal_policy_id(
    *,
    snapshot_registry: object,
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
    candidate_policy_ids: Sequence[str],
) -> str | None:
    """Recommend a non-baseline focal policy for reporting from a resolved final policy set.

    The recommendation prefers policies with canonicalized dev-eval summaries and falls back to
    the newest training snapshot among the eligible candidates when summary coverage is missing.
    """

    snapshot_policies = snapshot_training_policies(snapshot_registry)
    normalized_summaries = canonicalize_dev_eval_summaries(
        normalize_dev_eval_summaries(dev_eval_summaries),
        snapshot_policies=snapshot_policies,
    )
    eligible_policy_ids = [str(policy_id) for policy_id in candidate_policy_ids if _is_focal_candidate(str(policy_id))]
    if not eligible_policy_ids:
        return None

    summarized_candidates = [
        normalized_summaries[policy_id] for policy_id in eligible_policy_ids if policy_id in normalized_summaries
    ]
    if summarized_candidates:
        return max(
            summarized_candidates,
            key=lambda summary: (
                float(summary.aggregate_score),
                len(summary.anchor_scores),
                *training_policy_tie_break(summary.policy_id),
                summary.policy_id,
            ),
        ).policy_id

    snapshot_policies_by_id = {policy.policy_id: policy for policy in snapshot_policies}
    parsed_candidates: list[TrainingPolicyId] = []
    seen_policy_ids: set[str] = set()
    for policy_id in eligible_policy_ids:
        parsed_policy = snapshot_policies_by_id.get(policy_id)
        if parsed_policy is None:
            parsed_policy = try_parse_training_policy(policy_id)
        if parsed_policy is None or parsed_policy.policy_id in seen_policy_ids:
            continue
        parsed_candidates.append(parsed_policy)
        seen_policy_ids.add(parsed_policy.policy_id)
    if parsed_candidates:
        return max(parsed_candidates, key=training_policy_sort_key).policy_id

    return eligible_policy_ids[0]


def _is_focal_candidate(policy_id: str) -> bool:
    if policy_id in {RANDOM_LEGAL_POLICY_ID, NO_LEAGUE_POLICY_ID, LEGACY_NO_LEAGUE_POLICY_ID}:
        return False
    return heuristic_public_profile_name_for_policy_id(policy_id) is None


__all__ = ["recommend_focal_policy_id"]
