"""Factorized packed-policy scoring mixin for the structured action head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.eval.heuristic_public import heuristic_public_scoring_profile
from weiss_rl.models.action_plans import (
    FactorizedConditionalLogProbs,
    FactorizedEvaluationResult,
    FactorizedLegalityPlan,
    build_factorized_legality_plan,
)
from weiss_rl.models.factorized_conditionals import FactorizedConditionalLogProbsMixin
from weiss_rl.models.tensor_ops import (
    derived_sample_seeds,
    factorized_local_row_indices,
    masked_entropy_from_log_probs,
    masked_log_softmax,
    scatter_factorized_row_values,
)

_FactorizedEvaluationResult = FactorizedEvaluationResult
_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan
_masked_log_softmax = masked_log_softmax
_masked_entropy_from_log_probs = masked_entropy_from_log_probs
_derived_sample_seeds = derived_sample_seeds
_factorized_local_row_indices = factorized_local_row_indices
_scatter_factorized_row_values = scatter_factorized_row_values


def _segment_max(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    if keys.numel() == 0:
        return out
    out.scatter_reduce_(0, keys.to(dtype=torch.long), values, reduce="amax", include_self=True)
    return out


def _segment_logsumexp(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    max_per = _segment_max(values, keys, int(num_segments))
    if keys.numel() == 0:
        return max_per
    long_keys = keys.to(dtype=torch.long)
    gathered_max = max_per.index_select(0, long_keys)
    shifted = torch.exp(values - gathered_max)
    sumexp = torch.zeros((int(num_segments),), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, long_keys, shifted)
    valid = torch.isfinite(max_per) & (sumexp > 0)
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    out[valid] = torch.log(sumexp[valid]) + max_per[valid]
    return out


def _sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # Resolve lazily through weiss_rl.model so the private sampling wrapper remains monkeypatchable.
    from weiss_rl import model as model_module

    return model_module._sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=sample_seeds,
        default_index=default_index,
        temperature=temperature,
    )


class StructuredFactorizedScoringMixin(FactorizedConditionalLogProbsMixin):
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
        action_logp = None
        if actions is not None:
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
