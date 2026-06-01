"""Packed row margin helpers for paired-swing replay losses."""

from __future__ import annotations

import torch
from torch import Tensor


def positive_vs_top_other_margin_by_row(
    *,
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    flat_positive_actions: Tensor,
    active_rows: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_indices = torch.nonzero(active_rows, as_tuple=False).reshape(-1)
    supported = torch.zeros_like(active_rows, dtype=torch.bool)
    margin_by_row = torch.full_like(active_rows.to(dtype=packed_logits.dtype), -torch.inf)
    positive_logp_by_row = torch.full_like(margin_by_row, -torch.inf)
    top_other_logp_by_row = torch.full_like(margin_by_row, -torch.inf)
    offsets = legal_offsets.to(device=packed_logits.device, dtype=torch.long)
    ids = legal_ids.to(device=packed_logits.device, dtype=torch.long)
    for row_index_tensor in row_indices:
        row_index = int(row_index_tensor.detach().cpu().item())
        start = int(offsets[row_index].detach().cpu().item())
        stop = int(offsets[row_index + 1].detach().cpu().item())
        if stop <= start + 1:
            continue
        row_ids = ids[start:stop]
        positive_action = flat_positive_actions[row_index]
        positive_matches = row_ids == positive_action
        if not bool(positive_matches.any().item()):
            continue
        row_logp = torch.log_softmax(packed_logits[start:stop], dim=0)
        positive_logp = row_logp[positive_matches].max()
        top_other_logp = row_logp.masked_fill(positive_matches, float("-inf")).max()
        if not bool(torch.isfinite(positive_logp).item()) or not bool(torch.isfinite(top_other_logp).item()):
            continue
        supported[row_index] = True
        positive_logp_by_row[row_index] = positive_logp
        top_other_logp_by_row[row_index] = top_other_logp
        margin_by_row[row_index] = positive_logp - top_other_logp
    return margin_by_row, supported, positive_logp_by_row, top_other_logp_by_row


__all__ = ["positive_vs_top_other_margin_by_row"]
