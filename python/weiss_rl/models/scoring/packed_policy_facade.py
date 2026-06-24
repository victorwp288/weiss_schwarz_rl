"""Packed legal-action facade methods for structured policy/value models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.scoring.packed_action_sampling import sample_packed_action_scores


class StructuredPackedPolicyFacadeMixin:
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
        actions, behavior_logp = sample_packed_action_scores(
            packed_logits,
            torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long),
            torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long),
            sample_seeds.to(device=packed_logits.device, dtype=torch.long),
            pass_action_id=int(pass_action_id),
            temperature=temperature,
        )
        return actions, behavior_logp, value, next_seat_hidden


__all__ = ["StructuredPackedPolicyFacadeMixin"]
