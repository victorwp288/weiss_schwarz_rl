"""Data containers for structured and factorized model action scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.tensor_ops import packed_row_indices


@dataclass(frozen=True, slots=True)
class PackedScoringPlan:
    row_indices: Tensor
    family_ids: Tensor
    arg0: Tensor
    arg1: Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.family_ids.shape[0])

    def slice(self, start: int, end: int) -> PackedScoringPlan:
        return PackedScoringPlan(
            row_indices=self.row_indices[start:end],
            family_ids=self.family_ids[start:end],
            arg0=self.arg0[start:end],
            arg1=self.arg1[start:end],
        )


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
    same_family_arg0_logp: Tensor | None = None
    same_family_top_arg0: Tensor | None = None


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
    family_candidate_counts: Tensor
    family_plans: dict[int, FactorizedFamilyPlan]


def build_factorized_legality_plan(
    legal_actions: LegalActionBatch,
    *,
    device: torch.device,
    family_ids_by_action: Tensor,
    action_arg0: Tensor,
    action_arg1: Tensor,
    family_arg0_size: Tensor,
    family_arg1_size: Tensor,
    family_count: int,
) -> FactorizedLegalityPlan:
    if legal_actions.ids is None or legal_actions.offsets is None:
        raise ValueError("factorized structured policy requires packed legal ids and offsets")
    offsets = torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long)
    row_count = int(offsets.shape[0] - 1)
    if row_count < 0:
        raise ValueError("packed legal offsets must contain at least one row boundary")
    ids = torch.as_tensor(legal_actions.ids, device=device, dtype=torch.long)
    family_ids = family_ids_by_action.index_select(0, ids)
    arg0 = action_arg0.index_select(0, ids)
    arg1 = action_arg1.index_select(0, ids)
    row_indices = packed_row_indices(offsets)
    family_count_i = int(family_count)
    family_candidate_counts_flat = torch.zeros((row_count * family_count_i,), device=device, dtype=torch.long)
    if row_indices.numel() > 0:
        family_indices = row_indices * family_count_i + family_ids.to(dtype=torch.long)
        family_candidate_counts_flat.index_add_(0, family_indices, torch.ones_like(family_indices))
    family_candidate_counts = family_candidate_counts_flat.view(row_count, family_count_i)
    family_mask = family_candidate_counts > 0
    family_plans: dict[int, FactorizedFamilyPlan] = {}
    for family_id in range(family_count_i):
        family_candidate_mask = family_ids == int(family_id)
        if not bool(family_candidate_mask.any().item()):
            continue
        family_candidate_rows = row_indices[family_candidate_mask].to(dtype=torch.long)
        family_rows = torch.unique_consecutive(family_candidate_rows)
        arg0_size = int(family_arg0_size[family_id].item())
        arg0_mask: Tensor | None = None
        arg1_mask: Tensor | None = None
        if arg0_size > 0:
            local_row_indices = torch.searchsorted(family_rows, family_candidate_rows)
            family_arg0 = arg0[family_candidate_mask].to(dtype=torch.long)
            arg0_mask = torch.zeros((int(family_rows.shape[0]), arg0_size), device=device, dtype=torch.bool)
            valid_arg0 = family_arg0 >= 0
            if bool(valid_arg0.any().item()):
                arg0_mask[local_row_indices[valid_arg0], family_arg0[valid_arg0]] = True
            arg1_size = int(family_arg1_size[family_id].item())
            if arg1_size > 0:
                family_arg1 = arg1[family_candidate_mask].to(dtype=torch.long)
                valid_arg1 = valid_arg0 & (family_arg1 >= 0)
                arg1_mask = torch.zeros(
                    (int(family_rows.shape[0]), arg0_size, arg1_size),
                    device=device,
                    dtype=torch.bool,
                )
                if bool(valid_arg1.any().item()):
                    flat_index = (
                        local_row_indices[valid_arg1] * (arg0_size * arg1_size)
                        + family_arg0[valid_arg1] * arg1_size
                        + family_arg1[valid_arg1]
                    )
                    arg1_mask.view(-1)[flat_index] = True
        family_plans[family_id] = FactorizedFamilyPlan(
            row_indices=family_rows,
            arg0_mask=arg0_mask,
            arg1_mask=arg1_mask,
        )
    return FactorizedLegalityPlan(
        row_count=row_count,
        family_mask=family_mask,
        family_candidate_counts=family_candidate_counts,
        family_plans=family_plans,
    )
