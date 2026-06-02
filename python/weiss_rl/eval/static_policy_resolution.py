"""Static non-learned eval policy resolution."""

from __future__ import annotations

from collections.abc import Mapping

from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import (
    RANDOM_LEGAL_POLICY_ID,
    heuristic_public_profile_name_for_policy_id,
)
from weiss_rl.eval.policies.types import ResolvedEvalPolicy


def resolve_static_eval_policy(
    *,
    policy_id: str,
    spec_bundle: Mapping[str, object] | None,
) -> ResolvedEvalPolicy | None:
    if policy_id == RANDOM_LEGAL_POLICY_ID:
        return ResolvedEvalPolicy(policy_id=policy_id, kind="random_legal")

    heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
    if heuristic_profile is None:
        return None
    if spec_bundle is None:
        raise RuntimeError(f"Resolving {policy_id} requires the loaded simulator spec bundle")
    return ResolvedEvalPolicy(
        policy_id=policy_id,
        kind="heuristic_public",
        heuristic_policy=HeuristicPublicPolicy.from_spec_bundle(
            spec_bundle,
            scoring_profile=heuristic_profile,
        ),
    )


__all__ = ["resolve_static_eval_policy"]
