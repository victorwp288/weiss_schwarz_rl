"""Public legal-action scoring methods for the structured policy head."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.backbone.tensor_ops import negative_logits_fill_value
from weiss_rl.models.scoring.packed_legal_tensors import require_packed_legal_tensors
from weiss_rl.public_heuristic.profiles import heuristic_public_scoring_profile


class StructuredLegalActionScoringMixin:
    action_dim: int

    def score_legal_actions(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        resolved_state_repr, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )

        masked = torch.full(
            (latent.shape[0], self.action_dim),
            negative_logits_fill_value(latent.dtype),
            device=latent.device,
            dtype=latent.dtype,
        )
        if legal_actions is None:
            candidate_ids = torch.arange(self.action_dim, device=latent.device, dtype=torch.long)
            for row_index in range(latent.shape[0]):
                row_scores = self._score_candidates(
                    resolved_state_repr[row_index].unsqueeze(0),
                    torch.zeros((candidate_ids.shape[0],), device=latent.device, dtype=torch.long),
                    candidate_ids,
                    resolved_context,
                    scoring_mode=scoring_mode,
                )
                masked[row_index, candidate_ids] = row_scores.to(dtype=masked.dtype)
            return masked

        if legal_actions.ids is not None and legal_actions.offsets is not None:
            packed = require_packed_legal_tensors(
                legal_actions,
                device=latent.device,
                row_count=int(latent.shape[0]),
                missing_message="legal_actions must contain either packed ids or a mask",
            )
            row_scores = self.score_packed_candidates(
                latent,
                obs=obs,
                legal_actions=legal_actions,
                observation_context=resolved_context,
                state_repr=resolved_state_repr,
                scoring_mode=scoring_mode,
            )
            if row_scores.numel() > 0:
                lengths = packed.offsets[1:] - packed.offsets[:-1]
                row_indices = torch.repeat_interleave(
                    torch.arange(latent.shape[0], device=latent.device, dtype=torch.long),
                    lengths,
                )
                masked[row_indices, packed.ids] = row_scores.to(dtype=masked.dtype)
            return masked

        if legal_actions.mask is None:
            raise ValueError("legal_actions must contain either packed ids or a mask")
        legal_mask = torch.as_tensor(legal_actions.mask, device=latent.device, dtype=torch.bool)
        if legal_mask.ndim == 3 and legal_mask.shape[0] == 1:
            legal_mask = legal_mask[0]
        if legal_mask.ndim != 2 or legal_mask.shape[0] != latent.shape[0] or legal_mask.shape[1] != self.action_dim:
            raise ValueError("legal mask must have shape (batch, action) or (1, batch, action)")
        row_indices, candidate_ids = torch.nonzero(legal_mask, as_tuple=True)
        if candidate_ids.numel() > 0:
            row_scores = self._score_candidates_chunked(
                resolved_state_repr,
                row_indices.to(dtype=torch.long),
                candidate_ids.to(dtype=torch.long),
                resolved_context,
            )
            masked[row_indices, candidate_ids] = row_scores.to(dtype=masked.dtype)
        return masked

    def score_packed_candidates(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("score_packed_candidates requires packed legal ids and offsets")
        packed = require_packed_legal_tensors(
            legal_actions,
            device=latent.device,
            row_count=int(latent.shape[0]),
            missing_message="score_packed_candidates requires packed legal ids and offsets",
        )
        resolved_state_repr, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )
        if packed.candidate_count == 0:
            return latent.new_zeros((0,))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=packed.ids,
            offsets=packed.offsets,
            candidate_meta=packed.meta,
        )
        return self._score_packed_candidates_chunked(
            resolved_state_repr,
            scoring_plan,
            resolved_context,
            scoring_mode=scoring_mode,
        )

    def score_packed_public_heuristic_candidates(
        self,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None or legal_actions.meta is None:
            raise ValueError(
                "score_packed_public_heuristic_candidates requires packed legal ids, offsets, and metadata"
            )
        obs_batch = torch.as_tensor(obs)
        if obs_batch.ndim != 2:
            raise ValueError("score_packed_public_heuristic_candidates expects obs to be 2D (rows, observation)")
        resolved_profile = heuristic_public_scoring_profile(scoring_profile)
        resolved_context = (
            dict(observation_context)
            if observation_context is not None
            else self._encode_observation_context(obs_batch)
        )
        packed = require_packed_legal_tensors(
            legal_actions,
            device=obs_batch.device,
            row_count=int(obs_batch.shape[0]),
            require_meta=True,
            missing_message="score_packed_public_heuristic_candidates requires packed legal ids, offsets, and metadata",
        )
        if packed.candidate_count == 0:
            return obs_batch.new_zeros((0,))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=packed.ids,
            offsets=packed.offsets,
            candidate_meta=packed.meta,
        )
        return self._score_packed_public_heuristic_chunked(
            scoring_plan,
            resolved_context,
            dtype=obs_batch.dtype,
            scoring_profile=resolved_profile,
        )

    def forward(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        return self.score_legal_actions(
            latent,
            obs=obs,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )


__all__ = ["StructuredLegalActionScoringMixin"]
