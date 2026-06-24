"""Diagnostic summaries for factorized structured-policy outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import FactorizedConditionalLogProbs, FactorizedLegalityPlan
from weiss_rl.models.backbone.tensor_ops import factorized_local_row_indices

_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan
_factorized_local_row_indices = factorized_local_row_indices


class StructuredFactorizedDiagnosticsMixin:
    """Builds factorized top-action and same-family teacher-supervision summaries."""

    def _factorized_top_action_ids(
        self: Any,
        *,
        plan: _FactorizedLegalityPlan,
        family_log_probs: Tensor,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
    ) -> Tensor:
        row_count = int(plan.row_count)
        family_count = int(family_log_probs.shape[-1])
        best_family_action_ids = torch.full(
            (row_count, family_count),
            -1,
            device=family_log_probs.device,
            dtype=torch.long,
        )
        best_family_conditional_logp = torch.full_like(family_log_probs, -torch.inf)
        for family_id, family_plan in plan.family_plans.items():
            family_rows = family_plan.row_indices.to(dtype=torch.long)
            if family_rows.numel() == 0:
                continue
            family_kind = int(self._family_arg_kind[int(family_id)].item())
            if family_kind == 0:
                best_family_action_ids[family_rows, family_id] = int(
                    self._family_noarg_action_ids[int(family_id)].item()
                )
                best_family_conditional_logp[family_rows, family_id] = 0.0
                continue
            arg0_entry = arg0_log_probs.get(int(family_id))
            if arg0_entry is None:
                continue
            row_arg0_log_probs = arg0_entry.log_probs
            if family_kind in {1, 5, 6}:
                best_arg0_logp, best_arg0 = row_arg0_log_probs.max(dim=1)
                resolved_ids = self._one_arg_action_ids[int(family_id)].to(
                    device=family_log_probs.device, dtype=torch.long
                )
                best_family_action_ids[family_rows, family_id] = resolved_ids.index_select(0, best_arg0)
                best_family_conditional_logp[family_rows, family_id] = best_arg0_logp
                continue
            arg1_entry = arg1_log_probs.get(int(family_id))
            if arg1_entry is None:
                continue
            joint_log_probs = row_arg0_log_probs.unsqueeze(-1) + arg1_entry.log_probs
            flat_joint = joint_log_probs.reshape(joint_log_probs.shape[0], -1)
            best_joint_logp, best_joint = flat_joint.max(dim=1)
            arg1_size = int(joint_log_probs.shape[-1])
            best_arg0 = best_joint // arg1_size
            best_arg1 = best_joint % arg1_size
            resolved_ids = self._two_arg_action_ids[int(family_id)].to(device=family_log_probs.device, dtype=torch.long)
            best_family_action_ids[family_rows, family_id] = resolved_ids[best_arg0, best_arg1]
            best_family_conditional_logp[family_rows, family_id] = best_joint_logp
        total_logp = torch.where(
            best_family_action_ids >= 0,
            family_log_probs + best_family_conditional_logp,
            torch.full_like(family_log_probs, -torch.inf),
        )
        best_family = total_logp.argmax(dim=1)
        return best_family_action_ids.gather(1, best_family.unsqueeze(1)).squeeze(1)

    def _factorized_same_family_action_stats(
        self: Any,
        *,
        plan: _FactorizedLegalityPlan,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        reference_actions: Tensor,
        reference_families: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_ids = reference_actions.reshape(-1).to(device=self._family_ids.device, dtype=torch.long)
        family_ids = reference_families.reshape(-1).to(device=self._family_ids.device, dtype=torch.long)
        row_count = int(plan.row_count)
        same_family_action_logp = torch.full(
            (row_count,),
            -torch.inf,
            device=self._family_ids.device,
            dtype=dtype,
        )
        same_family_top_action_ids = torch.full(
            (row_count,),
            -1,
            device=self._family_ids.device,
            dtype=torch.long,
        )
        same_family_arg0_logp = torch.full(
            (row_count,),
            -torch.inf,
            device=self._family_ids.device,
            dtype=dtype,
        )
        same_family_top_arg0 = torch.full(
            (row_count,),
            -1,
            device=self._family_ids.device,
            dtype=torch.long,
        )
        if action_ids.numel() != row_count or family_ids.numel() != row_count or row_count == 0:
            return same_family_action_logp, same_family_top_action_ids, same_family_arg0_logp, same_family_top_arg0
        valid_rows = (
            (action_ids >= 0)
            & (action_ids < self.action_dim)
            & (family_ids >= 0)
            & (family_ids < plan.family_mask.shape[1])
        )
        if not bool(valid_rows.any().item()):
            return same_family_action_logp, same_family_top_action_ids, same_family_arg0_logp, same_family_top_arg0
        clamped_families = torch.clamp(family_ids, min=0, max=max(int(plan.family_mask.shape[1]) - 1, 0))
        valid_rows = valid_rows & plan.family_mask.gather(1, clamped_families.unsqueeze(1)).squeeze(1)
        if not bool(valid_rows.any().item()):
            return same_family_action_logp, same_family_top_action_ids, same_family_arg0_logp, same_family_top_arg0
        valid_row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(1)
        valid_action_ids = action_ids.index_select(0, valid_row_indices)
        valid_family_ids = family_ids.index_select(0, valid_row_indices)
        valid_action_family_ids = self._family_ids.index_select(0, valid_action_ids)
        valid_action_arg0 = self._action_arg0.index_select(0, valid_action_ids)
        valid_action_arg1 = self._action_arg1.index_select(0, valid_action_ids)
        for family_id in torch.unique(valid_family_ids, sorted=True).tolist():
            family_rows = valid_family_ids == int(family_id)
            if not bool(family_rows.any().item()):
                continue
            row_indices = valid_row_indices[family_rows]
            row_action_ids = valid_action_ids[family_rows]
            row_action_family_ids = valid_action_family_ids[family_rows]
            row_action_arg0 = valid_action_arg0[family_rows]
            row_action_arg1 = valid_action_arg1[family_rows]
            family_kind = int(self._family_arg_kind[int(family_id)].item())
            if family_kind == 0:
                resolved_id = int(self._family_noarg_action_ids[int(family_id)].item())
                same_family_top_action_ids[row_indices] = resolved_id
                supported = row_action_ids == resolved_id
                if bool(supported.any().item()):
                    same_family_action_logp[row_indices[supported]] = 0.0
                continue
            arg0_entry = arg0_log_probs.get(int(family_id))
            if arg0_entry is None:
                continue
            local_row_indices = _factorized_local_row_indices(arg0_entry.row_indices, row_indices)
            row_arg0_log_probs = arg0_entry.log_probs.index_select(0, local_row_indices)
            row_arg0_mask = arg0_entry.mask.index_select(0, local_row_indices)
            row_top_arg0 = row_arg0_log_probs.argmax(dim=1)
            same_family_top_arg0[row_indices] = row_top_arg0
            if family_kind in {1, 5, 6}:
                resolved_ids = self._one_arg_action_ids[int(family_id)].to(device=row_indices.device, dtype=torch.long)
                same_family_top_action_ids[row_indices] = resolved_ids.index_select(0, row_top_arg0)
                supported = (row_action_family_ids == int(family_id)) & (row_action_arg0 >= 0)
                if bool(supported.any().item()):
                    gather_arg0 = torch.clamp(row_action_arg0, min=0)
                    supported = supported & row_arg0_mask.gather(1, gather_arg0.unsqueeze(1)).squeeze(1)
                if bool(supported.any().item()):
                    supported_arg0 = row_action_arg0[supported]
                    selected_arg0_logp = row_arg0_log_probs[supported].gather(1, supported_arg0.unsqueeze(1)).squeeze(1)
                    same_family_action_logp[row_indices[supported]] = selected_arg0_logp
                    same_family_arg0_logp[row_indices[supported]] = selected_arg0_logp
                continue
            arg1_entry = arg1_log_probs.get(int(family_id))
            if arg1_entry is None:
                continue
            row_arg1_log_probs = arg1_entry.log_probs.index_select(0, local_row_indices)
            row_arg1_mask = arg1_entry.mask.index_select(0, local_row_indices)
            joint_log_probs = row_arg0_log_probs.unsqueeze(-1) + row_arg1_log_probs
            flat_joint = joint_log_probs.reshape(joint_log_probs.shape[0], -1)
            top_joint = flat_joint.argmax(dim=1)
            arg1_size = int(joint_log_probs.shape[-1])
            top_arg0 = top_joint // arg1_size
            top_arg1 = top_joint % arg1_size
            resolved_ids = self._two_arg_action_ids[int(family_id)].to(device=row_indices.device, dtype=torch.long)
            same_family_top_action_ids[row_indices] = resolved_ids[top_arg0, top_arg1]
            supported = (row_action_family_ids == int(family_id)) & (row_action_arg0 >= 0) & (row_action_arg1 >= 0)
            if bool(supported.any().item()):
                gather_arg0 = torch.clamp(row_action_arg0, min=0)
                gather_arg1 = torch.clamp(row_action_arg1, min=0)
                supported = (
                    supported
                    & row_arg1_mask[
                        torch.arange(row_indices.shape[0], device=row_indices.device, dtype=torch.long),
                        gather_arg0,
                        gather_arg1,
                    ]
                )
            if bool(supported.any().item()):
                supported_arg0 = row_action_arg0[supported]
                supported_arg1 = row_action_arg1[supported]
                supported_rows = torch.arange(
                    row_indices.shape[0],
                    device=row_indices.device,
                    dtype=torch.long,
                )[supported]
                selected_arg0_logp = row_arg0_log_probs[supported].gather(1, supported_arg0.unsqueeze(1)).squeeze(1)
                same_family_arg0_logp[row_indices[supported]] = selected_arg0_logp
                same_family_action_logp[row_indices[supported]] = (
                    selected_arg0_logp + row_arg1_log_probs[supported_rows, supported_arg0, supported_arg1]
                )
        return same_family_action_logp, same_family_top_action_ids, same_family_arg0_logp, same_family_top_arg0


__all__ = ["StructuredFactorizedDiagnosticsMixin"]
