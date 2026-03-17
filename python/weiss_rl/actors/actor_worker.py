"""Actor worker scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask


def actor_behavior_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def actor_behavior_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


@dataclass(slots=True)
class ActorWorker:
    actor_id: int

    def run_once(self) -> None:
        """Single rollout-collection hook for the actor loop."""
        return None
