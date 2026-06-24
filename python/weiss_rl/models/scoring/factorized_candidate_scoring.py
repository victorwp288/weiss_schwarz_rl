"""Packed candidate log-probability assembly for the factorized head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.actions.action_plans import FactorizedConditionalLogProbs, FactorizedLegalityPlan
from weiss_rl.models.scoring.factorized_math import _factorized_local_row_indices, _segment_logsumexp
from weiss_rl.public_heuristic.profiles import heuristic_public_scoring_profile

_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan


class StructuredFactorizedCandidateScoringMixin:
    """Scores simulator-provided legal candidates from factorized family/argument scores."""

    def _factorized_candidate_log_probs(
        self: Any,
        *,
        legal_actions: LegalActionBatch,
        row_states: Tensor,
        resolved_context: Mapping[str, Tensor],
        plan: _FactorizedLegalityPlan,
        family_log_probs: Tensor,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        scoring_mode: str,
    ) -> Tensor:
        ids = torch.as_tensor(legal_actions.ids, device=row_states.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=row_states.device, dtype=torch.long)
        if ids.numel() == 0:
            return row_states.new_zeros((0,))
        row_indices = torch.repeat_interleave(
            torch.arange(plan.row_count, device=row_states.device, dtype=torch.long),
            offsets[1:] - offsets[:-1],
        )
        candidate_families = self._family_ids.index_select(0, ids).to(dtype=torch.long)
        candidate_arg0 = self._action_arg0.index_select(0, ids).to(dtype=torch.long)
        candidate_arg1 = self._action_arg1.index_select(0, ids).to(dtype=torch.long)
        candidate_logp = family_log_probs[row_indices, candidate_families]
        for family_id, arg0_entry in arg0_log_probs.items():
            family_mask = candidate_families == int(family_id)
            if not bool(family_mask.any().item()):
                continue
            candidate_positions = torch.nonzero(family_mask, as_tuple=False).squeeze(1)
            family_rows = row_indices.index_select(0, candidate_positions)
            local_row_indices = _factorized_local_row_indices(arg0_entry.row_indices, family_rows)
            arg0_indices = candidate_arg0.index_select(0, candidate_positions)
            valid_arg0 = arg0_indices >= 0
            if not bool(valid_arg0.any().item()):
                candidate_logp[candidate_positions] = -torch.inf
                continue
            valid_positions = candidate_positions[valid_arg0]
            valid_local_rows = local_row_indices[valid_arg0]
            valid_arg0 = arg0_indices[valid_arg0]
            candidate_logp[valid_positions] = (
                candidate_logp[valid_positions] + arg0_entry.log_probs[valid_local_rows, valid_arg0]
            )
            arg1_entry = arg1_log_probs.get(int(family_id))
            if arg1_entry is None:
                continue
            arg1_indices = candidate_arg1.index_select(0, valid_positions)
            valid_arg1 = arg1_indices >= 0
            if not bool(valid_arg1.any().item()):
                candidate_logp[valid_positions] = -torch.inf
                continue
            invalid_arg1_positions = valid_positions[~valid_arg1]
            if invalid_arg1_positions.numel() > 0:
                candidate_logp[invalid_arg1_positions] = -torch.inf
            valid_positions = valid_positions[valid_arg1]
            valid_local_rows = valid_local_rows[valid_arg1]
            valid_arg0 = valid_arg0[valid_arg1]
            valid_arg1_indices = arg1_indices[valid_arg1]
            candidate_logp[valid_positions] = (
                candidate_logp[valid_positions] + arg1_entry.log_probs[valid_local_rows, valid_arg0, valid_arg1_indices]
            )
        public_bias_scale = self._public_heuristic_logit_bias_scale_for(scoring_mode)
        if public_bias_scale > 0.0:
            candidate_meta = (
                None
                if legal_actions.meta is None
                else torch.as_tensor(legal_actions.meta, device=row_states.device, dtype=torch.long)
            )
            public_plan = self._build_packed_scoring_plan(
                candidate_ids=ids,
                offsets=offsets,
                candidate_meta=candidate_meta,
            )
            public_scores = self._score_packed_public_heuristic_chunked(
                public_plan,
                resolved_context,
                dtype=candidate_logp.dtype,
                scoring_profile=heuristic_public_scoring_profile("base"),
            )
            candidate_logp = candidate_logp + public_scores.to(dtype=candidate_logp.dtype) * float(public_bias_scale)
            row_log_z = _segment_logsumexp(candidate_logp, row_indices, plan.row_count)
            candidate_logp = candidate_logp - row_log_z.index_select(0, row_indices)
        return candidate_logp
