"""Factorized packed-policy facade methods for structured models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.action_plans import FactorizedEvaluationResult
from weiss_rl.models.packed_action_sampling import sample_packed_action_scores


class StructuredFactorizedPolicyFacadeMixin:
    def _factorized_packed_action_log_probs_with_context(
        self: Any,
        recurrent_output: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        state_repr: Tensor,
        observation_context: Mapping[str, Tensor],
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> Tensor:
        action_log_probs = self.policy_head.factorized_packed_action_log_probs(
            recurrent_output,
            obs=obs,
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        action_log_probs = self._apply_opponent_context_packed_candidate_residual_to_log_probs(
            action_log_probs,
            legal_actions,
            state_repr,
            opponent_context_index,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        return self._apply_opponent_context_packed_action_bias_to_log_probs(
            action_log_probs,
            legal_actions,
            opponent_context_index,
        )

    def _packed_log_prob_policy_stats(
        self: Any,
        packed_log_probs: Tensor,
        *,
        legal_actions: LegalActionBatch,
        actions: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("packed policy stats require packed legal ids and offsets")
        if packed_log_probs.ndim != 1:
            raise ValueError(f"packed_log_probs must be 1D, got shape {tuple(packed_log_probs.shape)}")
        ids = torch.as_tensor(legal_actions.ids, device=packed_log_probs.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=packed_log_probs.device, dtype=torch.long)
        row_count = int(offsets.numel() - 1)
        lengths = offsets[1:] - offsets[:-1]
        if int(ids.numel()) != int(packed_log_probs.numel()) or int(lengths.sum().item()) != int(
            packed_log_probs.numel()
        ):
            raise ValueError("packed legal ids/offsets must align with packed log-probs")
        row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_log_probs.device), lengths)
        entropy = torch.zeros((row_count,), device=packed_log_probs.device, dtype=packed_log_probs.dtype)
        if int(packed_log_probs.numel()) > 0:
            safe_log_probs = torch.where(
                torch.isfinite(packed_log_probs),
                packed_log_probs,
                torch.zeros_like(packed_log_probs),
            )
            entropy.scatter_add_(0, row_indices, -(torch.exp(packed_log_probs) * safe_log_probs))
        top_action_ids = torch.full((row_count,), -1, device=packed_log_probs.device, dtype=torch.long)
        if int(packed_log_probs.numel()) > 0:
            row_max = torch.full((row_count,), -torch.inf, device=packed_log_probs.device, dtype=packed_log_probs.dtype)
            row_max.scatter_reduce_(0, row_indices, packed_log_probs, reduce="amax", include_self=True)
            max_for_candidate = row_max.index_select(0, row_indices)
            sentinel = torch.iinfo(torch.long).max
            top_candidates = torch.where(
                packed_log_probs == max_for_candidate,
                ids,
                torch.full_like(ids, sentinel),
            )
            top_min = torch.full((row_count,), sentinel, device=packed_log_probs.device, dtype=torch.long)
            top_min.scatter_reduce_(0, row_indices, top_candidates, reduce="amin", include_self=True)
            top_action_ids = torch.where(top_min == sentinel, top_action_ids, top_min)

        action_logp = None
        if actions is not None:
            flat_actions = actions.reshape(-1).to(device=packed_log_probs.device, dtype=torch.long)
            if int(flat_actions.numel()) != row_count:
                raise ValueError(f"actions must have length {row_count}, got {int(flat_actions.numel())}")
            action_logp = torch.full(
                (row_count,),
                -torch.inf,
                device=packed_log_probs.device,
                dtype=packed_log_probs.dtype,
            )
            if int(packed_log_probs.numel()) > 0:
                candidate_actions = flat_actions.index_select(0, row_indices)
                selected_values = torch.where(
                    ids == candidate_actions,
                    packed_log_probs,
                    torch.full_like(packed_log_probs, -torch.inf),
                )
                action_logp.scatter_reduce_(0, row_indices, selected_values, reduce="amax", include_self=True)
        return action_logp, entropy, top_action_ids

    def _factorized_contextual_packed_policy_stats(
        self: Any,
        recurrent_output: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        state_repr: Tensor,
        observation_context: Mapping[str, Tensor],
        actions: Tensor | None = None,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor, Tensor] | None:
        if not self._has_opponent_context_packed_adjustment(
            opponent_context_index,
            row_count=int(recurrent_output.shape[0]),
            device=recurrent_output.device,
        ):
            return None
        packed_log_probs = self._factorized_packed_action_log_probs_with_context(
            recurrent_output,
            obs=obs,
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
            opponent_context_index=opponent_context_index,
        )
        return self._packed_log_prob_policy_stats(
            packed_log_probs,
            legal_actions=legal_actions,
            actions=actions,
        )

    def evaluate_factorized_sequence_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        actions: Tensor | None = None,
        same_family_reference_actions: Tensor | None = None,
        same_family_reference_families: Tensor | None = None,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> FactorizedEvaluationResult:
        recurrent_flat, state_repr, observation_context, values, _seat_hidden = self.forward_trunk_sequence_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )
        head_result = self.policy_head.evaluate_factorized_packed(
            recurrent_flat,
            obs=obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
            legal_actions=legal_actions,
            actions=None if actions is None else actions.reshape(-1),
            same_family_reference_actions=(
                None if same_family_reference_actions is None else same_family_reference_actions.reshape(-1)
            ),
            same_family_reference_families=(
                None if same_family_reference_families is None else same_family_reference_families.reshape(-1)
            ),
            state_repr=state_repr,
            observation_context=observation_context,
        )
        context_index_flat = (
            None
            if opponent_context_index is None
            else torch.as_tensor(opponent_context_index, device=recurrent_flat.device, dtype=torch.long).reshape(-1)
        )
        contextual_stats = self._factorized_contextual_packed_policy_stats(
            recurrent_flat,
            obs=obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
            legal_actions=legal_actions,
            actions=None if actions is None else actions.reshape(-1),
            state_repr=state_repr,
            observation_context=observation_context,
            opponent_context_index=context_index_flat,
        )
        action_logp = head_result.action_logp
        entropy = head_result.entropy
        top_action_ids = head_result.top_action_ids
        if contextual_stats is not None:
            contextual_action_logp, entropy, top_action_ids = contextual_stats
            action_logp = contextual_action_logp
        return FactorizedEvaluationResult(
            values=values,
            action_logp=None if action_logp is None else action_logp.reshape(obs.shape[0], obs.shape[1]),
            entropy=None if entropy is None else entropy.reshape(obs.shape[0], obs.shape[1]),
            family_log_probs=head_result.family_log_probs.reshape(
                obs.shape[0], obs.shape[1], head_result.family_log_probs.shape[-1]
            ),
            play_slot_log_probs=(
                None
                if head_result.play_slot_log_probs is None
                else head_result.play_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.play_slot_log_probs.shape[-1],
                )
            ),
            move_source_log_probs=(
                None
                if head_result.move_source_log_probs is None
                else head_result.move_source_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.move_source_log_probs.shape[-1],
                )
            ),
            move_slot_log_probs=(
                None
                if head_result.move_slot_log_probs is None
                else head_result.move_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.move_slot_log_probs.shape[-1],
                )
            ),
            attack_slot_log_probs=(
                None
                if head_result.attack_slot_log_probs is None
                else head_result.attack_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.attack_slot_log_probs.shape[-1],
                )
            ),
            attack_type_log_probs=(
                None
                if head_result.attack_type_log_probs is None
                else head_result.attack_type_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.attack_type_log_probs.shape[-1],
                )
            ),
            top_action_ids=(None if top_action_ids is None else top_action_ids.reshape(obs.shape[0], obs.shape[1])),
            same_family_action_logp=(
                None
                if head_result.same_family_action_logp is None
                else head_result.same_family_action_logp.reshape(obs.shape[0], obs.shape[1])
            ),
            same_family_top_action_ids=(
                None
                if head_result.same_family_top_action_ids is None
                else head_result.same_family_top_action_ids.reshape(obs.shape[0], obs.shape[1])
            ),
            same_family_arg0_logp=(
                None
                if head_result.same_family_arg0_logp is None
                else head_result.same_family_arg0_logp.reshape(obs.shape[0], obs.shape[1])
            ),
            same_family_top_arg0=(
                None
                if head_result.same_family_top_arg0 is None
                else head_result.same_family_top_arg0.reshape(obs.shape[0], obs.shape[1])
            ),
        )

    def sample_factorized_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
        temperature: float = 1.0,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        recurrent_output, state_repr, observation_context, value, next_seat_hidden = (
            self.forward_trunk_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                opponent_context_index=opponent_context_index,
            )
        )
        obs_batch = self._require_observation_batch(obs)
        if self._has_opponent_context_packed_adjustment(
            opponent_context_index,
            row_count=int(recurrent_output.shape[0]),
            device=recurrent_output.device,
        ):
            if legal_actions.ids is None or legal_actions.offsets is None:
                raise ValueError("sample_factorized_packed_seat_aware requires packed ids and offsets")
            packed_log_probs = self._factorized_packed_action_log_probs_with_context(
                recurrent_output,
                obs=obs_batch,
                legal_actions=legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
                opponent_context_index=opponent_context_index,
            )
            actions, behavior_logp = sample_packed_action_scores(
                packed_log_probs,
                torch.as_tensor(legal_actions.ids, device=packed_log_probs.device, dtype=torch.long),
                torch.as_tensor(legal_actions.offsets, device=packed_log_probs.device, dtype=torch.long),
                sample_seeds.to(device=packed_log_probs.device, dtype=torch.long),
                pass_action_id=int(pass_action_id),
                temperature=temperature,
            )
        else:
            actions, behavior_logp = self.policy_head.sample_factorized_packed(
                recurrent_output,
                obs=obs_batch,
                legal_actions=legal_actions,
                sample_seeds=sample_seeds,
                pass_action_id=pass_action_id,
                temperature=temperature,
                state_repr=state_repr,
                observation_context=observation_context,
            )
        return actions, behavior_logp, value, next_seat_hidden

    def factorized_packed_action_log_probs_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        recurrent_output, state_repr, observation_context, value, next_seat_hidden = (
            self.forward_trunk_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                opponent_context_index=opponent_context_index,
            )
        )
        action_log_probs = self._factorized_packed_action_log_probs_with_context(
            recurrent_output,
            obs=self._require_observation_batch(obs),
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
            opponent_context_index=opponent_context_index,
        )
        return action_log_probs, value, next_seat_hidden


__all__ = ["StructuredFactorizedPolicyFacadeMixin"]
