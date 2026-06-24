"""Factorized packed-policy scoring mixin for the structured action head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.actions.action_plans import (
    FactorizedConditionalLogProbs,
    FactorizedEvaluationResult,
    FactorizedLegalityPlan,
    build_factorized_legality_plan,
)
from weiss_rl.models.scoring.factorized_candidate_scoring import StructuredFactorizedCandidateScoringMixin
from weiss_rl.models.scoring.factorized_conditionals import FactorizedConditionalLogProbsMixin
from weiss_rl.models.scoring.factorized_diagnostics import StructuredFactorizedDiagnosticsMixin
from weiss_rl.models.scoring.factorized_evaluation_parts import StructuredFactorizedEvaluationPartsMixin
from weiss_rl.models.scoring.factorized_math import (
    _masked_log_softmax,
)
from weiss_rl.models.scoring.factorized_sampling import StructuredFactorizedSamplingMixin

_FactorizedEvaluationResult = FactorizedEvaluationResult
_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan


class StructuredFactorizedScoringMixin(
    StructuredFactorizedSamplingMixin,
    StructuredFactorizedDiagnosticsMixin,
    StructuredFactorizedEvaluationPartsMixin,
    StructuredFactorizedCandidateScoringMixin,
    FactorizedConditionalLogProbsMixin,
):
    """Factorized log-probability helpers used by `_StructuredLegalActionHead`."""

    def _factorized_row_chunk_size(self: Any, row_states: Tensor) -> int:
        if row_states.device.type != "cuda":
            return 0
        return (
            int(self._factorized_learner_row_chunk_size)
            if torch.is_grad_enabled()
            else int(self._factorized_actor_row_chunk_size)
        )

    def _dot_product_log_probs(
        self: Any,
        query: Tensor,
        candidate_repr: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if candidate_repr.ndim != 3 or mask.ndim != 2:
            raise ValueError("candidate_repr must be 3D and mask must be 2D")
        if candidate_repr.shape[:2] != mask.shape:
            raise ValueError("candidate_repr and mask must agree on row and candidate dimensions")
        if candidate_repr.shape[0] == 0:
            return candidate_repr.new_zeros((0, candidate_repr.shape[1]))
        logits = (candidate_repr.to(dtype=query.dtype) * query.unsqueeze(1)).sum(dim=-1)
        return _masked_log_softmax(logits, mask)

    def _family_condition_input(self: Any, row_states: Tensor, *, family_id: int) -> Tensor:
        family_ids = torch.full(
            (row_states.shape[0],),
            int(family_id),
            device=row_states.device,
            dtype=torch.long,
        )
        family_embed = self.family_embedding(family_ids).to(dtype=row_states.dtype)
        return torch.cat([row_states, family_embed], dim=1)

    def _build_factorized_legality_plan(
        self: Any,
        legal_actions: LegalActionBatch,
        *,
        device: torch.device,
    ) -> _FactorizedLegalityPlan:
        return build_factorized_legality_plan(
            legal_actions,
            device=device,
            family_ids_by_action=self._family_ids,
            action_arg0=self._action_arg0,
            action_arg1=self._action_arg1,
            family_arg0_size=self._family_arg0_size,
            family_arg1_size=self._family_arg1_size,
            family_count=int(self._family_arg_kind.shape[0]),
        )

    def _family_log_probs(
        self: Any, row_states: Tensor, family_mask: Tensor, family_candidate_counts: Tensor
    ) -> Tensor:
        family_logits = self.family_head(row_states) + self.family_bias.to(
            device=row_states.device,
            dtype=row_states.dtype,
        )
        candidate_count_prior = torch.log(
            family_candidate_counts.to(device=row_states.device, dtype=row_states.dtype).clamp_min(1.0)
        )
        family_logits = family_logits + torch.where(family_mask, candidate_count_prior, torch.zeros_like(family_logits))
        return _masked_log_softmax(family_logits, family_mask)

    def _factorized_distributions(
        self: Any,
        row_states: Tensor,
        *,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor],
    ) -> tuple[
        _FactorizedLegalityPlan,
        Tensor,
        dict[int, _FactorizedConditionalLogProbs],
        dict[int, _FactorizedConditionalLogProbs],
    ]:
        plan = self._build_factorized_legality_plan(legal_actions, device=row_states.device)
        family_log_probs = self._family_log_probs(row_states, plan.family_mask, plan.family_candidate_counts)
        arg0_log_probs: dict[int, _FactorizedConditionalLogProbs] = {}
        arg1_log_probs: dict[int, _FactorizedConditionalLogProbs] = {}
        hand_ids = observation_context["hand_ids"].to(device=row_states.device, dtype=torch.long)
        self_stage_context = observation_context["self_stage_context"].to(
            device=row_states.device, dtype=row_states.dtype
        )
        for family_id, family_plan in plan.family_plans.items():
            kind = int(self._family_arg_kind[family_id].item())
            if kind == 0:
                continue
            family_rows = family_plan.row_indices
            arg0_mask = family_plan.arg0_mask
            if arg0_mask is None:
                continue
            family_row_states = row_states.index_select(0, family_rows)
            if kind in {1, 2}:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._hand_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        hand_ids=hand_ids.index_select(0, family_rows),
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            elif kind in {3, 4, 5}:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._slot_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            elif kind == 6:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._index_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            arg1_mask = family_plan.arg1_mask
            if arg1_mask is None:
                continue
            if family_id == self._play_character_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._play_arg1_log_probs(
                        family_row_states,
                        hand_ids=hand_ids.index_select(0, family_rows),
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
            elif family_id == self._main_move_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._move_arg1_log_probs(
                        family_row_states,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
            elif family_id == self._attack_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._attack_arg1_log_probs(
                        family_row_states,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
        return plan, family_log_probs, arg0_log_probs, arg1_log_probs

    def evaluate_factorized_packed(
        self: Any,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        actions: Tensor | None = None,
        same_family_reference_actions: Tensor | None = None,
        same_family_reference_families: Tensor | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
    ) -> _FactorizedEvaluationResult:
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
        row_count = int(row_states.shape[0])
        (
            entropy,
            play_slot_log_probs,
            move_source_log_probs,
            move_slot_log_probs,
            attack_slot_log_probs,
            attack_type_log_probs,
        ) = self._factorized_entropy_and_projection_summaries(
            row_count=row_count,
            plan=plan,
            family_log_probs=family_log_probs,
            arg0_log_probs=arg0_log_probs,
            arg1_log_probs=arg1_log_probs,
        )
        action_logp = self._factorized_selected_action_logp(
            actions=actions,
            row_states=row_states,
            family_log_probs=family_log_probs,
            arg0_log_probs=arg0_log_probs,
            arg1_log_probs=arg1_log_probs,
        )
        top_action_ids = self._factorized_top_action_ids(
            plan=plan,
            family_log_probs=family_log_probs,
            arg0_log_probs=arg0_log_probs,
            arg1_log_probs=arg1_log_probs,
        )
        same_family_action_logp = None
        same_family_top_action_ids = None
        same_family_arg0_logp = None
        same_family_top_arg0 = None
        if same_family_reference_actions is not None and same_family_reference_families is not None:
            (
                same_family_action_logp,
                same_family_top_action_ids,
                same_family_arg0_logp,
                same_family_top_arg0,
            ) = self._factorized_same_family_action_stats(
                plan=plan,
                arg0_log_probs=arg0_log_probs,
                arg1_log_probs=arg1_log_probs,
                reference_actions=same_family_reference_actions,
                reference_families=same_family_reference_families,
                dtype=row_states.dtype,
            )
        return _FactorizedEvaluationResult(
            values=row_states.new_zeros((row_count,)),
            action_logp=action_logp,
            entropy=entropy,
            family_log_probs=family_log_probs,
            play_slot_log_probs=play_slot_log_probs,
            move_source_log_probs=move_source_log_probs,
            move_slot_log_probs=move_slot_log_probs,
            attack_slot_log_probs=attack_slot_log_probs,
            attack_type_log_probs=attack_type_log_probs,
            top_action_ids=top_action_ids,
            same_family_action_logp=same_family_action_logp,
            same_family_top_action_ids=same_family_top_action_ids,
            same_family_arg0_logp=same_family_arg0_logp,
            same_family_top_arg0=same_family_top_arg0,
        )

    def factorized_packed_action_log_probs(
        self: Any,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        """Return factorized log-probability for every packed legal candidate."""
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("factorized packed action log-probs require packed legal ids and offsets")
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
        return self._factorized_candidate_log_probs(
            legal_actions=legal_actions,
            row_states=row_states,
            resolved_context=resolved_context,
            plan=plan,
            family_log_probs=family_log_probs,
            arg0_log_probs=arg0_log_probs,
            arg1_log_probs=arg1_log_probs,
            scoring_mode=scoring_mode,
        )
