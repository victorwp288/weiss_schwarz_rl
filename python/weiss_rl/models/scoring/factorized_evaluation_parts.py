"""Evaluation summaries for factorized structured-policy outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import FactorizedConditionalLogProbs, FactorizedLegalityPlan
from weiss_rl.models.scoring.factorized_math import (
    _factorized_local_row_indices,
    _masked_entropy_from_log_probs,
    _scatter_factorized_row_values,
)

_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan


class StructuredFactorizedEvaluationPartsMixin:
    """Assembles factorized entropy, projection summaries, and selected-action logp."""

    def _factorized_entropy_and_projection_summaries(
        self: Any,
        *,
        row_count: int,
        plan: _FactorizedLegalityPlan,
        family_log_probs: Tensor,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        entropy = _masked_entropy_from_log_probs(family_log_probs, plan.family_mask)
        play_slot_log_probs = None
        move_source_log_probs = None
        move_slot_log_probs = None
        attack_slot_log_probs = None
        attack_type_log_probs = None
        for family_id, arg0_entry in arg0_log_probs.items():
            family_rows = arg0_entry.row_indices
            family_prob = torch.exp(family_log_probs.index_select(0, family_rows)[:, family_id])
            arg0_entropy = _masked_entropy_from_log_probs(arg0_entry.log_probs, arg0_entry.mask)
            entropy.index_add_(0, family_rows, family_prob * arg0_entropy)
            arg1_entry = arg1_log_probs.get(family_id)
            if arg1_entry is None or plan.family_plans[family_id].arg1_mask is None:
                if family_id == self._attack_family_id:
                    attack_slot_log_probs = _scatter_factorized_row_values(
                        row_count,
                        family_rows,
                        arg0_entry.log_probs,
                    )
                continue
            arg1_entropy = _masked_entropy_from_log_probs(
                arg1_entry.log_probs.reshape(-1, arg1_entry.log_probs.shape[-1]),
                arg1_entry.mask.reshape(-1, arg1_entry.mask.shape[-1]),
            ).reshape(arg1_entry.log_probs.shape[0], arg1_entry.log_probs.shape[1])
            arg0_probs = torch.where(
                arg0_entry.mask, torch.exp(arg0_entry.log_probs), torch.zeros_like(arg0_entry.log_probs)
            )
            entropy.index_add_(0, family_rows, family_prob * (arg0_probs * arg1_entropy).sum(dim=1))
            if family_id == self._play_character_family_id:
                play_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
            elif family_id == self._main_move_family_id:
                move_source_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    arg0_entry.log_probs,
                )
                move_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
            elif family_id == self._attack_family_id:
                attack_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    arg0_entry.log_probs,
                )
                attack_type_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
        return (
            entropy,
            play_slot_log_probs,
            move_source_log_probs,
            move_slot_log_probs,
            attack_slot_log_probs,
            attack_type_log_probs,
        )

    def _factorized_selected_action_logp(
        self: Any,
        *,
        actions: Tensor | None,
        row_states: Tensor,
        family_log_probs: Tensor,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
    ) -> Tensor | None:
        if actions is None:
            return None
        flat_actions = actions.reshape(-1).to(device=row_states.device, dtype=torch.long)
        selected_family = self._family_ids.index_select(0, flat_actions).to(dtype=torch.long)
        selected_arg0 = self._action_arg0.index_select(0, flat_actions).to(dtype=torch.long)
        selected_arg1 = self._action_arg1.index_select(0, flat_actions).to(dtype=torch.long)
        action_logp = family_log_probs.gather(1, selected_family.unsqueeze(1)).squeeze(1)
        for family_id, arg0_entry in arg0_log_probs.items():
            family_rows = selected_family == int(family_id)
            if not bool(family_rows.any().item()):
                continue
            row_indices = torch.nonzero(family_rows, as_tuple=False).squeeze(1)
            local_row_indices = _factorized_local_row_indices(arg0_entry.row_indices, row_indices)
            arg0_indices = selected_arg0.index_select(0, row_indices)
            action_logp[row_indices] = action_logp[row_indices] + arg0_entry.log_probs.index_select(
                0, local_row_indices
            ).gather(
                1,
                arg0_indices.unsqueeze(1),
            ).squeeze(1)
            arg1_entry = arg1_log_probs.get(family_id)
            if arg1_entry is None:
                continue
            arg1_indices = selected_arg1.index_select(0, row_indices)
            action_logp[row_indices] = action_logp[row_indices] + arg1_entry.log_probs.index_select(
                0, local_row_indices
            ).gather(
                1,
                arg0_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, arg1_entry.log_probs.shape[-1]),
            ).squeeze(1).gather(1, arg1_indices.unsqueeze(1)).squeeze(1)
        return action_logp
