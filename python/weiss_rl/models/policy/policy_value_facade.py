"""Facade methods for the structured policy/value model."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.backbone.policy_value_trunk import StructuredPolicyValueTrunkMixin
from weiss_rl.models.scoring.factorized_facade import StructuredFactorizedPolicyFacadeMixin
from weiss_rl.models.scoring.packed_policy_facade import StructuredPackedPolicyFacadeMixin


class StructuredLegalPolicyValueFacadeMixin(
    StructuredPolicyValueTrunkMixin,
    StructuredPackedPolicyFacadeMixin,
    StructuredFactorizedPolicyFacadeMixin,
):
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
