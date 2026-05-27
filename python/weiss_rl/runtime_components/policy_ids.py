"""Shared runtime policy-id constants."""

from __future__ import annotations

from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID

MIRROR_OPPONENT_POLICY_ID = "latest_policy_mirror"
FIXED_OPPONENT_EXCLUSIONS = frozenset({NOLEAGUE_BASELINE_POLICY_ID})

__all__ = [
    "FIXED_OPPONENT_EXCLUSIONS",
    "MIRROR_OPPONENT_POLICY_ID",
    "NOLEAGUE_BASELINE_POLICY_ID",
]
