"""Learner action log-probability helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from weiss_rl.core.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.tensor_ops import segment_logsumexp, segment_max


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


def packed_action_logp_and_entropy(
    logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]
    row_count = flat_logits.shape[0]

    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()):
        raise ValueError("legal_offsets must end at len(legal_ids)")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")
    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")
    if bool((ids < 0).any().item()) or bool((ids >= action_space).any().item()):
        raise ValueError(f"packed legal ids must be in [0, {action_space})")

    selected_logp = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    entropy = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        row_ids = torch.repeat_interleave(non_empty_rows, non_empty_widths)
        legal_logits = flat_logits[row_ids, ids]
        segment_max_values = torch.segment_reduce(legal_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max_values, non_empty_widths)
        shifted = legal_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)
        entropy_terms = -(torch.exp(log_probs) * log_probs)
        entropy_non_empty = torch.segment_reduce(entropy_terms, reduce="sum", lengths=non_empty_widths)
        entropy[non_empty_rows] = entropy_non_empty

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(matches.to(dtype=flat_logits.dtype), reduce="sum", lengths=non_empty_widths)
        illegal_rows = match_counts != 1.0
        if bool(illegal_rows.any().item()):
            bad_position = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
            bad_row = int(non_empty_rows[bad_position].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        selected_logp[non_empty_rows] = selected_non_empty

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

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


def packed_selected_action_logp(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
    strict: bool = True,
) -> Tensor:
    if packed_logits.ndim != 1:
        raise ValueError("packed_logits must be 1D")
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    row_count = int(flat_actions.shape[0])
    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()) or int(ids.numel()) != int(packed_logits.numel()):
        raise ValueError("packed logits, ids, and offsets must align exactly")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")

    selected_logp = torch.full(
        (row_count,),
        float("-inf") if not strict else 0.0,
        device=packed_logits.device,
        dtype=packed_logits.dtype,
    )
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        row_ids = torch.repeat_interleave(non_empty_rows, non_empty_widths)
        segment_max_values = torch.segment_reduce(packed_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max_values, non_empty_widths)
        shifted = packed_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(
            matches.to(dtype=packed_logits.dtype), reduce="sum", lengths=non_empty_widths
        )
        illegal_rows = match_counts != 1.0
        if strict and bool(illegal_rows.any().item()):
            bad_position = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
            bad_row = int(non_empty_rows[bad_position].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        if strict:
            selected_logp[non_empty_rows] = selected_non_empty
        else:
            supported_rows = match_counts == 1.0
            if bool(supported_rows.any().item()):
                selected_logp[non_empty_rows[supported_rows]] = selected_non_empty[supported_rows]

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            if strict:
                raise ValueError("pass_action_id is required when legality contains empty rows")
        else:
            support_empty_rows = flat_actions[empty_rows] == int(pass_action_id)
            if strict and bool((~support_empty_rows).any().item()):
                row_index = int(
                    torch.nonzero(empty_rows & (flat_actions != int(pass_action_id)), as_tuple=False)[0].item()
                )
                action = int(flat_actions[row_index].item())
                raise ValueError(
                    f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
                )
            if bool(support_empty_rows.any().item()):
                empty_row_ids = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
                selected_logp[empty_row_ids[support_empty_rows]] = 0.0

    return selected_logp.reshape(actions.shape)


def packed_subset_action_logp_and_top_action(
    packed_view: PackedStructuredLegalView,
    actions: Tensor,
    *,
    candidate_mask: Tensor,
    strict: bool = True,
) -> tuple[Tensor, Tensor]:
    flat_actions = actions.reshape(-1).to(dtype=torch.long, device=packed_view.logits.device)
    row_count = int(flat_actions.shape[0])
    if row_count != int(packed_view.row_count):
        raise ValueError("actions must align with the packed row count")
    if candidate_mask.shape != packed_view.action_ids.shape:
        raise ValueError("candidate_mask must align 1:1 with packed action ids")

    selected = candidate_mask.to(device=packed_view.logits.device, dtype=torch.bool)
    selected_logp = torch.full(
        (row_count,),
        float("-inf") if not strict else 0.0,
        device=packed_view.logits.device,
        dtype=packed_view.logits.dtype,
    )
    top_action_ids = torch.full(
        (row_count,),
        -1,
        device=packed_view.logits.device,
        dtype=torch.long,
    )
    if not bool(selected.any().item()):
        return selected_logp.reshape(actions.shape), top_action_ids.reshape(actions.shape)

    row_indices = packed_view.row_indices[selected].to(dtype=torch.long)
    subset_logits = packed_view.logits[selected]
    subset_action_ids = packed_view.action_ids[selected].to(dtype=torch.long)
    row_log_z = segment_logsumexp(subset_logits, row_indices, row_count)
    log_probs = subset_logits - row_log_z.index_select(0, row_indices)

    repeated_actions = flat_actions.index_select(0, row_indices)
    matches = subset_action_ids == repeated_actions
    match_counts = torch.zeros((row_count,), device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    match_counts.scatter_add_(0, row_indices, matches.to(dtype=packed_view.logits.dtype))
    illegal_rows = match_counts != 1.0
    if strict and bool(illegal_rows.any().item()):
        bad_row = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        bad_action = int(flat_actions[bad_row].item())
        raise ValueError(f"illegal action {bad_action} for subset row {bad_row}")

    selected_non_empty = torch.zeros((row_count,), device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    selected_non_empty.scatter_add_(0, row_indices, torch.where(matches, log_probs, torch.zeros_like(log_probs)))
    if strict:
        row_has_candidates = torch.zeros((row_count,), device=packed_view.logits.device, dtype=torch.bool)
        row_has_candidates[row_indices] = True
        selected_logp[row_has_candidates] = selected_non_empty[row_has_candidates]
    else:
        supported_rows = match_counts == 1.0
        if bool(supported_rows.any().item()):
            selected_logp[supported_rows] = selected_non_empty[supported_rows]

    top_logits = segment_max(subset_logits, row_indices, row_count)
    top_matches = subset_logits >= (top_logits.index_select(0, row_indices) - 1.0e-6)
    top_action_ids.scatter_reduce_(
        0,
        row_indices,
        torch.where(top_matches, subset_action_ids, torch.full_like(subset_action_ids, -1)),
        reduce="amax",
        include_self=True,
    )
    return selected_logp.reshape(actions.shape), top_action_ids.reshape(actions.shape)


def packed_scores_action_logp_and_entropy(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if packed_logits.ndim != 1:
        raise ValueError("packed_logits must be 1D")
    selected_logp = packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
        strict=True,
    )
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    row_count = int(flat_actions.shape[0])
    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()) or int(ids.numel()) != int(packed_logits.numel()):
        raise ValueError("packed logits, ids, and offsets must align exactly")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")

    entropy = torch.zeros((row_count,), device=packed_logits.device, dtype=packed_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        segment_max_values = torch.segment_reduce(packed_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max_values, non_empty_widths)
        shifted = packed_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)
        entropy_terms = -(torch.exp(log_probs) * log_probs)
        entropy_non_empty = torch.segment_reduce(entropy_terms, reduce="sum", lengths=non_empty_widths)
        entropy[non_empty_rows] = entropy_non_empty

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


def packed_scores_family_entropy(
    packed_logits: Tensor,
    legal_offsets: Tensor,
    legal_action_meta: Tensor,
    *,
    row_shape: torch.Size | tuple[int, ...],
    family_count: int,
) -> Tensor:
    """Entropy over legal action-family probability masses for packed candidate scores."""

    if packed_logits.ndim != 1:
        raise ValueError("packed_logits must be 1D")
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    if offsets.ndim != 1:
        raise ValueError("legal_offsets must be 1D")
    row_count = int(offsets.numel()) - 1
    if row_count < 0:
        raise ValueError("legal_offsets must contain at least one offset")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(packed_logits.numel()):
        raise ValueError("legal_offsets must end at len(packed_logits)")
    if int(family_count) <= 0:
        raise ValueError("family_count must be positive")

    meta = legal_action_meta.to(device=packed_logits.device)
    if meta.ndim != 2 or int(meta.shape[0]) != int(packed_logits.numel()) or int(meta.shape[1]) < 1:
        raise ValueError("legal_action_meta must have shape (num_legal, meta_width >= 1)")
    family_ids = meta[:, 0].to(dtype=torch.long)
    if family_ids.numel() and (
        bool((family_ids < 0).any().item()) or bool((family_ids >= int(family_count)).any().item())
    ):
        raise ValueError("legal_action_meta contains family ids outside configured family_count")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")

    entropy = torch.zeros((row_count,), device=packed_logits.device, dtype=packed_logits.dtype)
    non_empty_rows = torch.nonzero(widths > 0, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return entropy.reshape(row_shape)

    non_empty_widths = widths[non_empty_rows]
    row_indices = torch.repeat_interleave(non_empty_rows, non_empty_widths)
    segment_max_values = torch.segment_reduce(packed_logits, reduce="max", lengths=non_empty_widths)
    repeated_max = torch.repeat_interleave(segment_max_values, non_empty_widths)
    shifted = packed_logits - repeated_max
    exp_shifted = torch.exp(shifted)
    segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
    repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
    candidate_probs = exp_shifted / repeated_sum

    flat_family_indices = (row_indices * int(family_count)) + family_ids
    family_probs = torch.zeros(
        (row_count * int(family_count),),
        device=packed_logits.device,
        dtype=packed_logits.dtype,
    )
    family_probs.scatter_add_(0, flat_family_indices, candidate_probs)
    family_probs = family_probs.reshape(row_count, int(family_count))
    safe_log_probs = torch.where(family_probs > 0.0, torch.log(family_probs), torch.zeros_like(family_probs))
    entropy = -(family_probs * safe_log_probs).sum(dim=1)
    return entropy.reshape(row_shape)
