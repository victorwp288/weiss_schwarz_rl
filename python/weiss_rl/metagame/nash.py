"""Nash-mixture solver scaffold."""

from __future__ import annotations

import numpy as np


def uniform_mixture(num_policies: int) -> np.ndarray:
    """Temporary deterministic fallback mixture."""
    if num_policies <= 0:
        raise ValueError("num_policies must be > 0")
    return np.full((num_policies,), 1.0 / num_policies, dtype=np.float64)
