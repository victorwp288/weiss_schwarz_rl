"""Factorized-policy helper containers for the structured action head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FactorizedEvaluationResult:
    values: Tensor
    action_logp: Tensor | None
    entropy: Tensor | None
    family_log_probs: Tensor
    play_slot_log_probs: Tensor | None
    move_source_log_probs: Tensor | None
    move_slot_log_probs: Tensor | None
    attack_slot_log_probs: Tensor | None
    attack_type_log_probs: Tensor | None
    top_action_ids: Tensor | None = None
    same_family_action_logp: Tensor | None = None
    same_family_top_action_ids: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FactorizedFamilyPlan:
    row_indices: Tensor
    arg0_mask: Tensor | None
    arg1_mask: Tensor | None


@dataclass(frozen=True, slots=True)
class FactorizedConditionalLogProbs:
    row_indices: Tensor
    log_probs: Tensor
    mask: Tensor


@dataclass(frozen=True, slots=True)
class FactorizedLegalityPlan:
    row_count: int
    family_mask: Tensor
    family_plans: dict[int, FactorizedFamilyPlan]


def factorized_local_row_indices(available_rows: Tensor, selected_rows: Tensor) -> Tensor:
    if selected_rows.numel() == 0:
        return selected_rows.new_zeros((0,), dtype=torch.long)
    if available_rows.numel() == 0:
        raise ValueError("factorized row lookup requires at least one available row")
    positions = torch.searchsorted(available_rows, selected_rows)
    if bool((positions >= available_rows.shape[0]).any().item()):
        raise ValueError("factorized row lookup exceeded available rows")
    matched_rows = available_rows.index_select(0, positions)
    if not bool(torch.equal(matched_rows, selected_rows)):
        raise ValueError("factorized row lookup requires selected rows to be legal for the chosen family")
    return positions


def scatter_factorized_row_values(
    row_count: int,
    row_indices: Tensor,
    values: Tensor,
    *,
    fill_value: float = -torch.inf,
) -> Tensor:
    output = values.new_full((row_count, *values.shape[1:]), fill_value)
    if row_indices.numel() > 0:
        output.index_copy_(0, row_indices.to(dtype=torch.long), values)
    return output
