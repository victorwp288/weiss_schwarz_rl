"""Policy execution helpers for replay inspection."""

from __future__ import annotations

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.model_action_surface import ModelActionSurfaceSettings, model_action_surface_batch_and_ids
from weiss_rl.eval.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.model import GLOBAL_ACTION_SPACE_SIZE
from weiss_rl.replay.inspection_policy_loading import LoadedReplayPolicy


def policy_action_surface_batch_and_ids(
    *,
    policy: LoadedReplayPolicy,
    stack: StackConfig,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    pass_action_id: int,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    """Mirror eval-time model-only action-surface guards for replay scoring."""

    if policy.model is None:
        return batch, legal_ids
    return model_action_surface_batch_and_ids(
        model=policy.model,
        batch=batch,
        legal_ids=legal_ids,
        settings=ModelActionSurfaceSettings.from_training_config(
            stack.config.training,
            pass_action_id=pass_action_id,
        ),
    )


def forward_policy(
    *,
    policy: LoadedReplayPolicy,
    batch: DecisionBoundaryBatch,
    seat_hidden: torch.Tensor | None,
    legal_ids: np.ndarray,
    opponent_context_index: int | None = None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    if policy.heuristic_policy is not None:
        chosen_action = int(
            policy.heuristic_policy.choose_action(np.asarray(batch.obs[0], dtype=np.float32), np.asarray(legal_ids))
        )
        logits = np.full((GLOBAL_ACTION_SPACE_SIZE,), -1.0e9, dtype=np.float32)
        logits[chosen_action] = 0.0
        return logits, seat_hidden

    if policy.model is None or seat_hidden is None:
        raise RuntimeError(f"Replay policy {policy.label!r} is missing a model or hidden state")

    device = torch.device("cpu")
    acting_seat = int(batch.actor[0])
    with torch.inference_mode():
        logits, next_seat_hidden = model_eval_logits_for_legal_ids(
            model=policy.model,
            batch=batch,
            current_seat=acting_seat,
            seat_hidden=seat_hidden,
            legal_ids=legal_ids,
            action_dim=GLOBAL_ACTION_SPACE_SIZE,
            device=device,
            opponent_context_index=opponent_context_index,
        )
    return logits, next_seat_hidden


__all__ = [
    "forward_policy",
    "policy_action_surface_batch_and_ids",
]
