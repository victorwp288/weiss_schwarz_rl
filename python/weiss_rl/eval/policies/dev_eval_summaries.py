"""Dev-eval summary validation and canonicalization for policy selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from weiss_rl.eval.policies.training_policy_ids import TrainingPolicyId, try_parse_training_policy


@dataclass(frozen=True, slots=True)
class DevEvalPolicySummary:
    policy_id: str
    aggregate_score: float
    anchor_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_scores", {key: float(value) for key, value in self.anchor_scores.items()})

    def mean_anchor_score(self, anchor_policy_ids: Sequence[str]) -> float:
        if not anchor_policy_ids:
            return self.aggregate_score
        missing = [policy_id for policy_id in anchor_policy_ids if policy_id not in self.anchor_scores]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"dev-eval summary for {self.policy_id!r} is missing anchor scores for: {missing_text}")
        total = sum(self.anchor_scores[policy_id] for policy_id in anchor_policy_ids)
        return total / len(anchor_policy_ids)


DevEvalSummaryLike = float | DevEvalPolicySummary


def normalize_dev_eval_summaries(
    dev_eval_summaries: Mapping[str, DevEvalSummaryLike],
) -> dict[str, DevEvalPolicySummary]:
    normalized: dict[str, DevEvalPolicySummary] = {}
    for policy_id, summary in dev_eval_summaries.items():
        if isinstance(summary, DevEvalPolicySummary):
            if summary.policy_id != policy_id:
                raise ValueError(
                    f"dev-eval summary key {policy_id!r} does not match embedded policy_id {summary.policy_id!r}"
                )
            normalized[policy_id] = summary
            continue
        if isinstance(summary, bool) or not isinstance(summary, (int, float)):
            raise TypeError(
                "dev_eval_summaries values must be floats or DevEvalPolicySummary instances, "
                f"got {type(summary).__name__} for {policy_id!r}"
            )
        normalized[policy_id] = DevEvalPolicySummary(policy_id=policy_id, aggregate_score=float(summary))
    return normalized


def canonicalize_dev_eval_summaries(
    dev_eval_summaries: Mapping[str, DevEvalPolicySummary],
    *,
    snapshot_policies: Sequence[TrainingPolicyId],
) -> dict[str, DevEvalPolicySummary]:
    registry_policy_id_by_key = {(policy.update, policy.version): policy.policy_id for policy in snapshot_policies}
    canonical: dict[str, DevEvalPolicySummary] = {}
    for policy_id, summary in dev_eval_summaries.items():
        canonical_policy_id = policy_id
        parsed = try_parse_training_policy(policy_id)
        if parsed is not None:
            canonical_policy_id = registry_policy_id_by_key.get((parsed.update, parsed.version), "")
            if not canonical_policy_id:
                continue
        existing = canonical.get(canonical_policy_id)
        candidate = DevEvalPolicySummary(
            policy_id=canonical_policy_id,
            aggregate_score=summary.aggregate_score,
            anchor_scores=summary.anchor_scores,
        )
        if existing is None:
            canonical[canonical_policy_id] = candidate
            continue
        if len(candidate.anchor_scores) > len(existing.anchor_scores):
            canonical[canonical_policy_id] = candidate
            continue
        if (
            len(candidate.anchor_scores) == len(existing.anchor_scores)
            and candidate.aggregate_score > existing.aggregate_score
        ):
            canonical[canonical_policy_id] = candidate
    return canonical


__all__ = [
    "DevEvalPolicySummary",
    "DevEvalSummaryLike",
    "canonicalize_dev_eval_summaries",
    "normalize_dev_eval_summaries",
]
