"""Observation and candidate-feature helpers for the structured policy head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from weiss_rl.core.observation_layout import ObservationSlice
from weiss_rl.models.actions.candidate_components import (
    CandidateComponentFamilyIds,
    resolve_candidate_components,
)
from weiss_rl.models.actions.candidate_projection import (
    project_candidate_sections,
    score_candidate_group,
)
from weiss_rl.models.backbone.tensor_ops import bucket_card_ids, optional_embedding
from weiss_rl.models.observations.feature_gathering import (
    gather_stage_features,
    gather_stage_features_for_rows,
    slot_component,
)
from weiss_rl.models.observations.observation_context import (
    encode_observation_context,
    encode_stage_slice,
    extract_card_vector,
    extract_header_scalar,
    extract_scalar_feature,
)


class StructuredHeadContextMixin:
    """Build state context and candidate features for structured action scoring."""

    def _build_state_representation(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        observation_context: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D (batch, hidden), got shape {tuple(latent.shape)}")
        if obs.ndim != 2 or obs.shape[0] != latent.shape[0]:
            raise ValueError("structured_v2 policy head requires obs with shape (batch, observation)")
        obs_batch = obs.to(device=latent.device, dtype=torch.float32)
        resolved_context = (
            self._encode_observation_context(obs_batch) if observation_context is None else dict(observation_context)
        )
        state_repr = self.state_projection(
            torch.cat(
                [
                    latent,
                    resolved_context["hand_summary"].to(dtype=latent.dtype),
                    resolved_context["self_stage_summary"].to(dtype=latent.dtype),
                    resolved_context["opponent_stage_summary"].to(dtype=latent.dtype),
                ],
                dim=1,
            )
        )
        return state_repr, resolved_context

    def _encode_observation_context(self, obs_batch: Tensor) -> dict[str, Tensor]:
        return encode_observation_context(
            obs_batch=obs_batch,
            observation_contract=self._observation_contract,
            slot_context_dim=int(self._slot_context_dim),
            stage_slot_count=int(self._stage_slot_count),
            card_representation=self._card_representation,
            hand_summary_projection=self.hand_summary_projection,
            slot_encoder=self.slot_encoder,
        )

    def _extract_scalar_feature(
        self,
        obs_batch: Tensor,
        slice_spec: ObservationSlice | None,
    ) -> Tensor:
        return extract_scalar_feature(obs_batch, slice_spec)

    def _extract_header_scalar(
        self,
        obs_batch: Tensor,
        index: int | None,
    ) -> Tensor:
        return extract_header_scalar(obs_batch, index)

    def _encode_stage_slice(
        self,
        obs_batch: Tensor,
        stage_slice: ObservationSlice | None,
    ) -> tuple[Tensor, Tensor]:
        return encode_stage_slice(
            obs_batch=obs_batch,
            stage_slice=stage_slice,
            observation_contract=self._observation_contract,
            stage_slot_count=int(self._stage_slot_count),
            slot_context_dim=int(self._slot_context_dim),
            card_representation=self._card_representation,
            slot_encoder=self.slot_encoder,
        )

    def _resolve_scoring_mode(self, scoring_mode: str) -> str:
        resolved_mode = str(scoring_mode).strip().lower()
        if resolved_mode == "auto":
            return "actor" if not torch.is_grad_enabled() else "learner"
        if resolved_mode not in {"actor", "learner"}:
            raise ValueError("scoring_mode must be one of: auto, actor, learner")
        return resolved_mode

    def _project_candidate_sections(
        self,
        *,
        feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
        numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
        constant_numeric_ones: Sequence[int] = (),
        scoring_mode: str = "auto",
    ) -> Tensor:
        return project_candidate_sections(
            candidate_projection=self.candidate_projection,
            numeric_feature_offset=self._numeric_feature_offset,
            feature_sections=feature_sections,
            numeric_sections=numeric_sections,
            constant_numeric_ones=constant_numeric_ones,
            scoring_mode=self._resolve_scoring_mode(scoring_mode),
        )

    def _score_candidate_group(
        self,
        row_states: Tensor,
        *,
        feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
        numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
        constant_numeric_ones: Sequence[int] = (),
        scoring_mode: str = "auto",
    ) -> Tensor:
        if row_states.numel() == 0:
            return row_states.new_zeros((0,))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        return score_candidate_group(
            row_states,
            candidate_projection=self.candidate_projection,
            joint_scorer=self.joint_scorer,
            numeric_feature_offset=self._numeric_feature_offset,
            feature_sections=feature_sections,
            numeric_sections=numeric_sections,
            constant_numeric_ones=constant_numeric_ones,
            scoring_mode=resolved_mode,
        )

    def _resolve_candidate_components(
        self,
        candidate_ids: Tensor,
        candidate_meta: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return resolve_candidate_components(
            candidate_ids,
            candidate_meta,
            family_ids_by_action=self._family_ids,
            hand_indices_by_action=self._hand_indices,
            stage_slots_by_action=self._stage_slots,
            from_slots_by_action=self._from_slots,
            to_slots_by_action=self._to_slots,
            attack_slots_by_action=self._attack_slots,
            attack_types_by_action=self._attack_types,
            generic_indices_by_action=self._generic_indices,
            meta_unused=int(self._meta_unused),
            family_ids=CandidateComponentFamilyIds(
                play_character=int(self._play_character_family_id),
                main_event=int(self._main_event_family_id),
                clock_from_hand=int(self._clock_from_hand_family_id),
                climax_play=int(self._climax_play_family_id),
                mulligan_select=int(self._mulligan_select_family_id),
                main_move=int(self._main_move_family_id),
                attack=int(self._attack_family_id),
                choice_select=int(self._choice_select_family_id),
                level_up=int(self._level_up_family_id),
                trigger_order=int(self._trigger_order_family_id),
            ),
        )

    def _gather_hand_embeddings_from_rows(
        self,
        hand_ids: Tensor,
        row_indices: Tensor,
        hand_indices: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if hand_ids.shape[1] == 0:
            return (
                torch.zeros_like(hand_indices, dtype=torch.bool),
                hand_ids.new_zeros((hand_indices.shape[0], self.card_embedding.embedding_dim), dtype=dtype),
            )
        hand_present = (hand_indices >= 0) & (hand_indices < hand_ids.shape[1])
        if not torch.any(hand_present):
            return (
                hand_present,
                hand_ids.new_zeros((hand_indices.shape[0], self.card_embedding.embedding_dim), dtype=dtype),
            )
        safe_rows = torch.where(hand_present, row_indices, torch.zeros_like(row_indices)).to(dtype=torch.long)
        safe_hand = torch.where(hand_present, hand_indices, torch.zeros_like(hand_indices)).to(dtype=torch.long)
        flat_indices = safe_rows * int(hand_ids.shape[1]) + safe_hand
        candidate_hand_ids = hand_ids.reshape(-1).index_select(0, flat_indices)
        hand_card_embeddings = self._card_representation(candidate_hand_ids, dtype=dtype)
        hand_position_embeddings = optional_embedding(self.hand_position_embedding, hand_indices).to(dtype=dtype)
        hand_card_embeddings = hand_card_embeddings + hand_position_embeddings
        return hand_present, hand_card_embeddings * hand_present.unsqueeze(1).to(dtype=dtype)

    def _gather_stage_features_for_rows(
        self,
        slot_contexts: Tensor,
        slot_numeric: Tensor,
        row_indices: Tensor,
        slot_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gather_stage_features_for_rows(
            slot_contexts,
            slot_numeric,
            row_indices,
            slot_indices,
            stage_slot_count=int(self._stage_slot_count),
        )

    def _card_representation(self, card_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
        bucketed_ids = bucket_card_ids(card_ids, vocab_size=self._card_vocab_size)
        learned = self.card_embedding(bucketed_ids).to(dtype=dtype)
        if self.card_feature_projection is None or self._card_static_features.numel() == 0:
            return learned
        flat_ids = bucketed_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        static_features = self._card_static_features.index_select(0, unique_ids)
        projected_unique = self.card_feature_projection(static_features.to(dtype=dtype))
        projected = projected_unique.index_select(0, inverse).reshape(
            *bucketed_ids.shape,
            projected_unique.shape[-1],
        )
        return learned + projected.to(dtype=dtype)

    def _gather_stage_features(
        self,
        slot_contexts: Tensor,
        slot_numeric: Tensor,
        slot_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gather_stage_features(
            slot_contexts,
            slot_numeric,
            slot_indices,
            stage_slot_count=int(self._stage_slot_count),
        )

    def _extract_card_vector(self, obs_batch: Tensor, observation_slice: ObservationSlice | None) -> Tensor:
        return extract_card_vector(obs_batch, observation_slice)

    def _slot_component(self, stage_values: Tensor, offset: int) -> Tensor:
        return slot_component(stage_values, int(offset))


__all__ = ["StructuredHeadContextMixin"]
