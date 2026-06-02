"""Shared output contract for central policy-routing phases."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CentralPolicyPhaseOutputs(NamedTuple):
    logits_steps: list[np.ndarray | None]
    value_steps: list[np.ndarray]
    action_steps: list[np.ndarray] | None
    logp_steps: list[np.ndarray] | None
