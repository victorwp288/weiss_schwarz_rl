from __future__ import annotations

import numpy as np


def sample_actions_for_policy(
    *,
    policy_name: str,
    legal_actions,
    base_seed: int,
    step_index: int,
) -> np.ndarray:
    """Return one action per env using the current legal-action helper.

    `legal_actions` is expected to be `batch.legal` from weiss_sim `ResetBatch` / `StepBatch`.
    """

    token = policy_name.strip().lower()

    if token == "first_legal":
        return legal_actions.first_legal()

    if token == "uniform_legal":
        # Step-dependent seed keeps the run deterministic while still varying actions.
        return legal_actions.sample_uniform(seed=int(base_seed) + int(step_index))

    raise ValueError(f"Unknown policy_name: {policy_name}")
