"""Learner action log-probability helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from weiss_rl.core.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.learners.packed_action_logp import (
    packed_action_logp_and_entropy as packed_action_logp_and_entropy,
)
from weiss_rl.learners.packed_action_logp import (
    packed_scores_action_logp_and_entropy as packed_scores_action_logp_and_entropy,
)
from weiss_rl.learners.packed_action_logp import (
    packed_scores_family_entropy as packed_scores_family_entropy,
)
from weiss_rl.learners.packed_action_logp import (
    packed_selected_action_logp as packed_selected_action_logp,
)
from weiss_rl.learners.packed_action_logp import (
    packed_subset_action_logp_and_top_action as packed_subset_action_logp_and_top_action,
)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
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


def masked_log_probs_and_entropy(logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")

    mask = legal_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_legal = mask.any(dim=1, keepdim=True)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    row_max = torch.where(has_legal, row_max, torch.zeros_like(row_max))

    shifted = torch.where(mask, logits - row_max, torch.full_like(logits, float("-inf")))
    exp_shifted = torch.where(mask, torch.exp(shifted), torch.zeros_like(logits))
    denom = exp_shifted.sum(dim=1, keepdim=True)
    safe_denom = torch.where(has_legal, denom, torch.ones_like(denom))
    log_probs = torch.where(mask, shifted - torch.log(safe_denom), torch.full_like(logits, float("-inf")))

    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=1)
    return log_probs, entropy


def masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_mask = legal_mask.reshape(-1, logits.shape[-1]).to(dtype=torch.bool)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]

    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")

    empty_rows = ~flat_mask.any(dim=1)
    row_actions = flat_actions.unsqueeze(1)
    action_is_legal = flat_mask.gather(dim=1, index=row_actions).squeeze(1)
    illegal_rows = (~empty_rows) & (~action_is_legal)
    if bool(illegal_rows.any().item()):
        row_index = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        action = int(flat_actions[row_index].item())
        raise ValueError(f"illegal action {action} for row {row_index}")

    log_probs, entropy = masked_log_probs_and_entropy(flat_logits, flat_mask)
    selected_logp = log_probs.gather(dim=1, index=row_actions).squeeze(1)

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )
        selected_logp = torch.where(empty_rows, torch.zeros_like(selected_logp), selected_logp)
        entropy = torch.where(empty_rows, torch.zeros_like(entropy), entropy)

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)
