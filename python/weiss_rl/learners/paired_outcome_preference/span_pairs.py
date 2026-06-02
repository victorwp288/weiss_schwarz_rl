"""Span-level pair aggregation for paired-outcome preference losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.pair_components import PreferencePairComponents


def span_preference_pair_components(
    *,
    current: Tensor,
    reference: Tensor,
    pair_ids: Tensor,
    roles: Tensor,
    valid: Tensor,
    group_ids: Tensor | None,
    pair_weight_rows: Tensor | None,
    unique_pair_ids: Tensor,
    aggregation: str,
    beta: float,
    dtype: torch.dtype,
) -> PreferencePairComponents:
    device = current.device
    margins: list[Tensor] = []
    pair_losses: list[Tensor] = []
    pair_weights: list[Tensor] = []
    pair_group_ids: list[int] = []
    current_pref_values: list[Tensor] = []
    current_rej_values: list[Tensor] = []
    incomplete_pair_count = 0
    for pair_id in unique_pair_ids:
        pair_mask = valid & (pair_ids == pair_id)
        preferred_mask = pair_mask & (roles == 1)
        rejected_mask = pair_mask & (roles == 0)
        if not bool(preferred_mask.any().item()) or not bool(rejected_mask.any().item()):
            incomplete_pair_count += 1
            continue
        cur_pref = _aggregate(current[preferred_mask], aggregation)
        ref_pref = _aggregate(reference[preferred_mask], aggregation)
        cur_rej = _aggregate(current[rejected_mask], aggregation)
        ref_rej = _aggregate(reference[rejected_mask], aggregation)
        margin = (cur_pref - ref_pref) - (cur_rej - ref_rej)
        margins.append(margin)
        pair_losses.append(-F.logsigmoid(margin * beta))
        if pair_weight_rows is None:
            pair_weights.append(torch.ones((), device=device, dtype=dtype))
        else:
            pair_weights.append(pair_weight_rows[pair_mask].mean().detach().to(device=device, dtype=dtype))
        if group_ids is not None:
            group_values = group_ids[pair_mask]
            pair_group_ids.append(int(group_values[0].detach().cpu().item()) if group_values.numel() else -1)
        current_pref_values.append(cur_pref)
        current_rej_values.append(cur_rej)
    return PreferencePairComponents(
        margins=margins,
        pair_losses=pair_losses,
        pair_weights=pair_weights,
        pair_group_ids=pair_group_ids,
        current_pref_values=current_pref_values,
        current_rej_values=current_rej_values,
        incomplete_pair_count=incomplete_pair_count,
    )


def _aggregate(values: Tensor, aggregation: str) -> Tensor:
    if aggregation == "sum":
        return values.sum()
    return values.mean()


__all__ = ["span_preference_pair_components"]
