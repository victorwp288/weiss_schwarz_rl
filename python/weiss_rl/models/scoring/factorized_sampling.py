"""Sampling helpers for the factorized structured policy head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.scoring.factorized_math import (
    _derived_sample_seeds,
    _factorized_local_row_indices,
    _sample_masked_log_probs,
)


class StructuredFactorizedSamplingMixin:
    """Samples concrete simulator actions from factorized family/argument scores."""

    def sample_factorized_packed(
        self: Any,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
        temperature: float = 1.0,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        row_states, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )
        plan, family_log_probs, arg0_log_probs, arg1_log_probs = self._factorized_distributions(
            row_states,
            legal_actions=legal_actions,
            observation_context=resolved_context,
        )
        family_actions, behavior_logp = _sample_masked_log_probs(
            family_log_probs,
            plan.family_mask,
            sample_seeds=sample_seeds.to(device=row_states.device, dtype=torch.long),
            default_index=max(self._pass_family_id, 0),
            temperature=temperature,
        )
        actions = torch.full((row_states.shape[0],), int(pass_action_id), device=row_states.device, dtype=torch.long)
        for family_id in range(int(self._family_arg_kind.shape[0])):
            family_rows = torch.nonzero(family_actions == int(family_id), as_tuple=False).squeeze(1)
            if family_rows.numel() == 0:
                continue
            kind = int(self._family_arg_kind[family_id].item())
            if kind == 0:
                resolved_ids = self._family_noarg_action_ids[family_id]
                actions[family_rows] = torch.where(
                    resolved_ids >= 0,
                    resolved_ids.to(device=row_states.device, dtype=torch.long).expand_as(family_rows),
                    torch.full_like(family_rows, int(pass_action_id), dtype=torch.long),
                )
                continue
            arg0_log_probs_family = arg0_log_probs.get(family_id)
            if arg0_log_probs_family is None:
                continue
            local_row_indices = _factorized_local_row_indices(arg0_log_probs_family.row_indices, family_rows)
            arg0_actions, arg0_logp = _sample_masked_log_probs(
                arg0_log_probs_family.log_probs.index_select(0, local_row_indices),
                arg0_log_probs_family.mask.index_select(0, local_row_indices),
                sample_seeds=_derived_sample_seeds(sample_seeds.index_select(0, family_rows), salt=0x9E3779B1),
                default_index=0,
                temperature=temperature,
            )
            behavior_logp[family_rows] = behavior_logp[family_rows] + arg0_logp
            if kind in {1, 5, 6}:
                resolved_ids = self._one_arg_action_ids[family_id].to(device=row_states.device, dtype=torch.long)
                action_ids = resolved_ids.index_select(0, arg0_actions)
                actions[family_rows] = torch.where(
                    action_ids >= 0,
                    action_ids,
                    torch.full_like(action_ids, int(pass_action_id)),
                )
                continue
            arg1_log_probs_family = arg1_log_probs.get(family_id)
            if arg1_log_probs_family is None:
                continue
            row_arg1_log_probs = arg1_log_probs_family.log_probs.index_select(0, local_row_indices)[
                torch.arange(family_rows.shape[0], device=row_states.device, dtype=torch.long),
                arg0_actions,
            ]
            row_arg1_mask = arg1_log_probs_family.mask.index_select(0, local_row_indices)[
                torch.arange(family_rows.shape[0], device=row_states.device, dtype=torch.long),
                arg0_actions,
            ]
            arg1_actions, arg1_logp = _sample_masked_log_probs(
                row_arg1_log_probs,
                row_arg1_mask,
                sample_seeds=_derived_sample_seeds(sample_seeds.index_select(0, family_rows), salt=0x85EBCA77),
                default_index=0,
                temperature=temperature,
            )
            behavior_logp[family_rows] = behavior_logp[family_rows] + arg1_logp
            resolved_ids = self._two_arg_action_ids[family_id].to(device=row_states.device, dtype=torch.long)
            action_ids = resolved_ids[arg0_actions, arg1_actions]
            actions[family_rows] = torch.where(
                action_ids >= 0,
                action_ids,
                torch.full_like(action_ids, int(pass_action_id)),
            )
        return actions, behavior_logp
