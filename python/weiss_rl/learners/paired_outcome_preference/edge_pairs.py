"""Time-aligned edge-mean pair aggregation for paired-outcome preference losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from weiss_rl.learners.paired_outcome_preference.pair_components import PreferencePairComponents


def edge_mean_preference_pair_components(
    *,
    current_action_logp: Tensor,
    reference_action_logp: Tensor,
    pair_ids: Tensor,
    roles: Tensor,
    valid: Tensor,
    group_ids: Tensor | None,
    pair_weight_rows: Tensor | None,
    unique_pair_ids: Tensor,
    beta: float,
    dtype: torch.dtype,
) -> PreferencePairComponents:
    if current_action_logp.ndim != 2:
        raise ValueError("paired outcome preference edge_mean aggregation requires 2D time-major tensors")
    device = current_action_logp.device
    current = current_action_logp.to(device=device, dtype=dtype)
    reference = reference_action_logp.to(device=device, dtype=dtype)
    pair_ids_2d = pair_ids.to(device=device, dtype=torch.long)
    roles_2d = roles.to(device=device, dtype=torch.long)
    valid_2d = valid.to(device=device, dtype=torch.bool)
    group_ids_2d = None if group_ids is None else group_ids.to(device=device, dtype=torch.long)
    weights_2d = None if pair_weight_rows is None else pair_weight_rows.to(device=device, dtype=dtype)

    margins: list[Tensor] = []
    edge_losses: list[Tensor] = []
    edge_weights: list[Tensor] = []
    edge_group_ids: list[int] = []
    current_pref_values: list[Tensor] = []
    current_rej_values: list[Tensor] = []
    incomplete_pair_count = 0
    time_steps = int(current.shape[0])
    for pair_id in unique_pair_ids:
        pair_mask = valid_2d & (pair_ids_2d == pair_id)
        preferred_mask = pair_mask & (roles_2d == 1)
        rejected_mask = pair_mask & (roles_2d == 0)
        if not bool(preferred_mask.any().item()) or not bool(rejected_mask.any().item()):
            incomplete_pair_count += 1
            continue
        pair_group_id = -1
        if group_ids_2d is not None:
            group_values = group_ids_2d[pair_mask]
            pair_group_id = int(group_values[0].detach().cpu().item()) if group_values.numel() else -1
        edge_count = 0
        for step in range(time_steps):
            preferred_step = preferred_mask[step]
            rejected_step = rejected_mask[step]
            if not bool(preferred_step.any().item()) or not bool(rejected_step.any().item()):
                continue
            cur_pref = current[step][preferred_step].mean()
            ref_pref = reference[step][preferred_step].mean()
            cur_rej = current[step][rejected_step].mean()
            ref_rej = reference[step][rejected_step].mean()
            margin = (cur_pref - ref_pref) - (cur_rej - ref_rej)
            margins.append(margin)
            edge_losses.append(-F.logsigmoid(margin * beta))
            if weights_2d is None:
                edge_weights.append(torch.ones((), device=device, dtype=dtype))
            else:
                edge_mask = preferred_step | rejected_step
                edge_weights.append(weights_2d[step][edge_mask].mean().detach().to(device=device, dtype=dtype))
            edge_group_ids.append(pair_group_id)
            current_pref_values.append(cur_pref)
            current_rej_values.append(cur_rej)
            edge_count += 1
        if edge_count <= 0:
            incomplete_pair_count += 1
    return PreferencePairComponents(
        margins=margins,
        pair_losses=edge_losses,
        pair_weights=edge_weights,
        pair_group_ids=edge_group_ids,
        current_pref_values=current_pref_values,
        current_rej_values=current_rej_values,
        incomplete_pair_count=incomplete_pair_count,
    )


__all__ = ["edge_mean_preference_pair_components"]
