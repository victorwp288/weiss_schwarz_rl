"""Result object for one simulator policy decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class EvalPolicyStep:
    """Chosen action plus optional model context needed by search overrides."""

    action: int
    next_seat_hidden: torch.Tensor | None
    logits: np.ndarray | None = None
    legal_ids_for_model: np.ndarray | None = None

    @property
    def has_model_surface(self) -> bool:
        return self.logits is not None and self.legal_ids_for_model is not None


__all__ = ["EvalPolicyStep"]
