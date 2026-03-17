from __future__ import annotations

import numpy as np

from weiss_rl.masking import MaskingAnomalyCounters, apply_empty_legal_action_fallback


def sample_actions_for_policy(
    *,
    policy_name: str,
    legal_actions,
    base_seed: int,
    step_index: int,
    counters: MaskingAnomalyCounters | None = None,
) -> np.ndarray:
    """Return one action per env using the current legal-action helper."""
    token = policy_name.strip().lower()

    if token == "first_legal":
        sampled_actions = legal_actions.first_legal()
    elif token in {"uniform_legal", "random_legal"}:
        sampled_actions = legal_actions.sample_uniform(seed=int(base_seed) + int(step_index))
    else:
        raise ValueError(f"Unknown policy_name: {policy_name}")

    legal_mask = getattr(legal_actions, "mask", None)
    if legal_mask is None:
        return sampled_actions

    _, adjusted_actions = apply_empty_legal_action_fallback(
        sampled_actions,
        legal_mask,
        counters=counters,
    )
    return adjusted_actions
