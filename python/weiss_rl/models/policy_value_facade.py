"""Facade methods for the structured policy/value model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.action_plans import FactorizedEvaluationResult

_FactorizedEvaluationResult = FactorizedEvaluationResult


def _sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # Resolve lazily through weiss_rl.model so the compatibility wrapper remains monkeypatchable.
    from weiss_rl import model as model_module

    return model_module._sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=pass_action_id,
        temperature=temperature,
    )


class StructuredLegalPolicyValueFacadeMixin:
    if TYPE_CHECKING:
        _compiled_trunk_packed_core: Any | None
        _compiled_trunk_sequence_core: Any | None
        _trunk_compile_last_error: str | None

    def encode(self: Any, obs: Tensor) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        if self._card_scalar_indices.numel() == 0:
            return self.encoder(obs_batch)
        prepared = obs_batch * self._encoder_input_keep_mask.to(device=obs_batch.device, dtype=obs_batch.dtype)
        return self.encoder(prepared)

    def forward(
        self: Any,
        obs: Tensor,
        hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_sequence_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        recurrent_flat, flat_obs_batch, seat_hidden, time_steps, batch_size = self._sequence_recurrent_outputs(
            obs,
            acting_seat,
            seat_hidden_state,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )
        logits_flat = self.policy_head.score_legal_actions(
            recurrent_flat,
            obs=flat_obs_batch,
            legal_actions=legal_actions,
        )
        logits_flat = self._apply_opponent_context_action_bias(
            logits_flat,
            None if opponent_context_index is None else opponent_context_index.reshape(-1),
        )
        value_flat = self.value_head(recurrent_flat).squeeze(-1)
        return (
            logits_flat.reshape(time_steps, batch_size, logits_flat.shape[-1]),
            value_flat.reshape(time_steps, batch_size),
            seat_hidden,
        )

    def forward_sequence_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "learner",
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        recurrent_flat, state_repr, observation_context, values, seat_hidden = self.forward_trunk_sequence_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )
        packed_logits = self.score_packed_legal_candidates(
            recurrent_flat,
            obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
            opponent_context_index=(None if opponent_context_index is None else opponent_context_index.reshape(-1)),
        )
        return packed_logits, values, seat_hidden

    def enable_trunk_compile(self: Any, *, mode: str = "reduce-overhead") -> Any:
        compiled_packed = self._compiled_trunk_packed_core
        compiled_sequence = self._compiled_trunk_sequence_core
        if compiled_packed is None:
            compiled_packed = torch.compile(
                self._forward_trunk_packed_core,
                mode=mode,
            )
        if compiled_sequence is None:
            compiled_sequence = torch.compile(
                self._forward_trunk_sequence_core,
                mode=mode,
            )
        self._compiled_trunk_packed_core = compiled_packed
        self._compiled_trunk_sequence_core = compiled_sequence
        return self

    def set_public_heuristic_logit_bias_scale(
        self: Any,
        value: float,
        *,
        actor_value: float | None = None,
    ) -> None:
        self.policy_head.set_public_heuristic_logit_bias_scales(
            learner_scale=float(value),
            actor_scale=None if actor_value is None else float(actor_value),
        )

    def get_public_heuristic_logit_bias_scale(self: Any, *, scoring_mode: str = "learner") -> float:
        return float(self.policy_head._public_heuristic_logit_bias_scale_for(scoring_mode))

    def advance_seat_hidden(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        _, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return next_seat_hidden

    def value_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, _next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return self.value_head(recurrent_output).squeeze(-1)

    def score_packed_legal_candidates(
        self: Any,
        recurrent_outputs: Tensor,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        *,
        state_repr: Tensor | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> Tensor:
        recurrent_batch = recurrent_outputs
        if recurrent_batch.ndim != 2:
            raise ValueError("recurrent_outputs must be 2D (rows, hidden)")
        obs_batch = self._require_observation_batch(obs)
        if legal_actions.ids is None or legal_actions.offsets is None or legal_actions.meta is None:
            raise ValueError("score_packed_legal_candidates requires packed ids, offsets, and metadata")
        packed_scores = self.policy_head.score_packed_candidates(
            recurrent_batch,
            obs=obs_batch,
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        if state_repr is not None:
            packed_scores = self._apply_opponent_context_packed_candidate_residual(
                packed_scores,
                legal_actions,
                state_repr,
                opponent_context_index,
                observation_context=observation_context,
                scoring_mode=scoring_mode,
            )
        return self._apply_opponent_context_packed_action_bias(
            packed_scores,
            legal_actions,
            opponent_context_index,
        )

    def score_packed_public_heuristic_candidates(
        self: Any,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        *,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        return self.policy_head.score_packed_public_heuristic_candidates(
            obs=obs_batch,
            legal_actions=legal_actions,
            observation_context=observation_context,
            scoring_profile=scoring_profile,
        )

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
    ) -> _FactorizedEvaluationResult:
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
        return _FactorizedEvaluationResult(
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

    def forward_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "actor",
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
        packed_logits = self.score_packed_legal_candidates(
            recurrent_output,
            self._require_observation_batch(obs),
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
            opponent_context_index=opponent_context_index,
        )
        return packed_logits, value, next_seat_hidden

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
            actions, behavior_logp = _sample_packed_action_scores(
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

    def sample_packed_seat_aware(
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
        packed_logits, value, next_seat_hidden = self.forward_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            legal_actions=legal_actions,
            scoring_mode="actor",
            opponent_context_index=opponent_context_index,
        )
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("sample_packed_seat_aware requires packed ids and offsets")
        actions, behavior_logp = _sample_packed_action_scores(
            packed_logits,
            torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long),
            torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long),
            sample_seeds.to(device=packed_logits.device, dtype=torch.long),
            pass_action_id=int(pass_action_id),
            temperature=temperature,
        )
        return actions, behavior_logp, value, next_seat_hidden

    def forward_trunk_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        trunk_forward = self._compiled_trunk_packed_core
        has_context = opponent_context_index is not None
        if trunk_forward is not None and not has_context:
            try:
                recurrent_output, obs_batch, value, next_seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_packed_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    opponent_context_index=opponent_context_index,
                )
        else:
            recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                obs,
                acting_seat,
                seat_hidden_state,
                opponent_context_index=opponent_context_index,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(recurrent_output, obs=obs_batch)
        return recurrent_output, state_repr, observation_context, value, next_seat_hidden

    def forward_trunk_sequence_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        time_steps = int(obs.shape[0])
        batch_size = int(obs.shape[1])
        trunk_forward = self._compiled_trunk_sequence_core
        has_resets = reset_before_step is not None and bool(torch.as_tensor(reset_before_step).any().item())
        has_context = opponent_context_index is not None
        if trunk_forward is not None and not has_resets and not has_context:
            try:
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_sequence_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    reset_before_step=reset_before_step,
                    opponent_context_index=opponent_context_index,
                )
        else:
            recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                obs,
                acting_seat,
                seat_hidden_state,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(
            recurrent_flat,
            obs=flat_obs_batch,
        )
        return recurrent_flat, state_repr, observation_context, value_flat.reshape(time_steps, batch_size), seat_hidden

    def _forward_trunk_packed_core(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return recurrent_output, obs_batch, value, next_seat_hidden

    def _forward_trunk_sequence_core(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        recurrent_flat, flat_obs_batch, seat_hidden, _time_steps, _batch_size = self._sequence_recurrent_outputs(
            obs,
            acting_seat,
            seat_hidden_state,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )
        value_flat = self.value_head(recurrent_flat).squeeze(-1)
        return recurrent_flat, flat_obs_batch, value_flat, seat_hidden

    def _sequence_recurrent_outputs(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        time_steps, batch_size, obs_dim = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        encoded_flat = self.encode(flat_obs)
        encoded = encoded_flat.reshape(time_steps, batch_size, encoded_flat.shape[-1])
        seat_hidden = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded[0],
        )
        reset_mask = None
        if reset_before_step is not None:
            reset_mask = torch.as_tensor(reset_before_step, device=encoded.device, dtype=torch.bool)
            if reset_mask.ndim != 2 or reset_mask.shape != obs.shape[:2]:
                raise ValueError("reset_before_step must be 2D (time, batch) with the same leading dimensions as obs")
        context_index = None
        if opponent_context_index is not None:
            context_index = torch.as_tensor(opponent_context_index, device=encoded.device, dtype=torch.long)
            if context_index.ndim != 2 or context_index.shape != obs.shape[:2]:
                raise ValueError(
                    "opponent_context_index must be 2D (time, batch) with the same leading dimensions as obs"
                )
        recurrent_steps: list[Tensor] = []
        for step_index, (step_encoded, step_seat) in enumerate(
            zip(encoded.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)
        ):
            if reset_mask is not None:
                step_reset = reset_mask[step_index]
                if bool(step_reset.any().item()):
                    reset_rows = torch.nonzero(step_reset, as_tuple=False).squeeze(1)
                    seat_hidden = seat_hidden.clone()
                    seat_hidden.index_copy_(
                        0,
                        reset_rows,
                        self.initial_seat_hidden(
                            int(reset_rows.numel()),
                            device=seat_hidden.device,
                            dtype=seat_hidden.dtype,
                            opponent_context_indices=(
                                None if context_index is None else context_index[step_index].index_select(0, reset_rows)
                            ),
                        ),
                    )
            recurrent_output, seat_hidden = self.recurrent_step_seat_aware(
                step_encoded,
                step_seat,
                seat_hidden,
            )
            recurrent_output = self._apply_opponent_context_recurrent_adapter(
                recurrent_output,
                None if context_index is None else context_index[step_index],
            )
            recurrent_steps.append(recurrent_output)
        recurrent = torch.stack(recurrent_steps, dim=0)
        recurrent_flat = recurrent.reshape(time_steps * batch_size, recurrent.shape[-1])
        return recurrent_flat, self._require_observation_batch(flat_obs), seat_hidden, time_steps, batch_size
