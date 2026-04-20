"""Backward-compatible re-export for the deterministic policy-set selector."""

from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    NO_LEAGUE_POLICY_ID,
    RANDOM_LEGAL_POLICY_ID,
    DevEvalPolicySummary,
    TrainingPolicyId,
    parse_training_policy_id,
    select_final_policy_set_deterministic_v1,
)

__all__ = [
    "DevEvalPolicySummary",
    "HEURISTIC_PUBLIC_POLICY_ID",
    "NO_LEAGUE_POLICY_ID",
    "RANDOM_LEGAL_POLICY_ID",
    "TrainingPolicyId",
    "parse_training_policy_id",
    "select_final_policy_set_deterministic_v1",
]
