"""Candidate feature helpers for the structured legal action head."""

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.observation_layout import ObservationSlice
from weiss_rl.structured_observation import bucket_card_ids as _bucket_card_ids
from weiss_rl.structured_observation import optional_embedding as _optional_embedding


class StructuredCandidateFeaturesMixin:
    def _resolve_candidate_components(
        self,
        candidate_ids: Tensor,
        candidate_meta: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if candidate_meta is None:
            return (
                self._family_ids.index_select(0, candidate_ids),
                self._hand_indices.index_select(0, candidate_ids),
                self._stage_slots.index_select(0, candidate_ids),
                self._from_slots.index_select(0, candidate_ids),
                self._to_slots.index_select(0, candidate_ids),
                self._attack_slots.index_select(0, candidate_ids),
                self._attack_types.index_select(0, candidate_ids),
                self._generic_indices.index_select(0, candidate_ids),
            )
        family_ids = candidate_meta[:, 0].to(dtype=torch.long)
        arg0 = candidate_meta[:, 1].to(dtype=torch.long)
        arg1 = candidate_meta[:, 2].to(dtype=torch.long)
        meta_unused = torch.full_like(arg0, self._meta_unused)
        arg0 = torch.where(arg0 == meta_unused, torch.full_like(arg0, -1), arg0)
        arg1 = torch.where(arg1 == meta_unused, torch.full_like(arg1, -1), arg1)

        hand_indices = torch.full_like(arg0, -1)
        hand_family_ids = (
            self._play_character_family_id,
            self._main_event_family_id,
            self._clock_from_hand_family_id,
            self._climax_play_family_id,
            self._mulligan_select_family_id,
        )
        for family_id in hand_family_ids:
            if family_id < 0:
                continue
            family_mask = family_ids == family_id
            hand_indices[family_mask] = arg0[family_mask]

        stage_slots = torch.full_like(arg0, -1)
        if self._play_character_family_id >= 0:
            play_mask = family_ids == self._play_character_family_id
            stage_slots[play_mask] = arg1[play_mask]

        from_slots = torch.full_like(arg0, -1)
        to_slots = torch.full_like(arg0, -1)
        if self._main_move_family_id >= 0:
            move_mask = family_ids == self._main_move_family_id
            from_slots[move_mask] = arg0[move_mask]
            to_slots[move_mask] = arg1[move_mask]

        attack_slots = torch.full_like(arg0, -1)
        attack_types = torch.full_like(arg0, -1)
        if self._attack_family_id >= 0:
            attack_mask = family_ids == self._attack_family_id
            attack_slots[attack_mask] = arg0[attack_mask]
            attack_types[attack_mask] = arg1[attack_mask]

        generic_indices = torch.full_like(arg0, -1)
        generic_family_ids = (
            self._choice_select_family_id,
            self._level_up_family_id,
            self._trigger_order_family_id,
        )
        for family_id in generic_family_ids:
            if family_id < 0:
                continue
            generic_mask = family_ids == family_id
            generic_indices[generic_mask] = arg0[generic_mask]

        return (
            family_ids,
            hand_indices,
            stage_slots,
            from_slots,
            to_slots,
            attack_slots,
            attack_types,
            generic_indices,
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
        hand_position_embeddings = _optional_embedding(self.hand_position_embedding, hand_indices).to(dtype=dtype)
        hand_card_embeddings = hand_card_embeddings + hand_position_embeddings
        return hand_present, hand_card_embeddings * hand_present.unsqueeze(1).to(dtype=dtype)

    def _gather_stage_features_for_rows(
        self,
        slot_contexts: Tensor,
        slot_numeric: Tensor,
        row_indices: Tensor,
        slot_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
        if not torch.any(valid):
            return (
                slot_contexts.new_zeros((slot_indices.shape[0], slot_contexts.shape[-1])),
                slot_numeric.new_zeros((slot_indices.shape[0], slot_numeric.shape[-1])),
            )
        safe_rows = torch.where(valid, row_indices, torch.zeros_like(row_indices)).to(dtype=torch.long)
        safe_slots = torch.where(valid, slot_indices, torch.zeros_like(slot_indices)).to(dtype=torch.long)
        flat_indices = safe_rows * self._stage_slot_count + safe_slots
        gathered_context = slot_contexts.reshape(-1, slot_contexts.shape[-1]).index_select(0, flat_indices)
        gathered_numeric = slot_numeric.reshape(-1, slot_numeric.shape[-1]).index_select(0, flat_indices)
        return (
            gathered_context * valid.unsqueeze(1).to(dtype=slot_contexts.dtype),
            gathered_numeric * valid.unsqueeze(1).to(dtype=slot_numeric.dtype),
        )

    def _card_representation(self, card_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
        bucketed_ids = _bucket_card_ids(card_ids, vocab_size=self._card_vocab_size)
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
        if slot_contexts.ndim == 3:
            valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
            safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
            context_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_contexts.shape[-1])
            numeric_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_numeric.shape[-1])
            gathered_context = torch.gather(slot_contexts, 1, context_index).squeeze(1)
            gathered_numeric = torch.gather(slot_numeric, 1, numeric_index).squeeze(1)
            return (
                gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
                gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
            )
        valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
        safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
        gathered_context = slot_contexts.index_select(0, safe_indices.to(dtype=torch.long))
        gathered_numeric = slot_numeric.index_select(0, safe_indices.to(dtype=torch.long))
        return (
            gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
            gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
        )

    def _extract_card_vector(self, obs_batch: Tensor, observation_slice: ObservationSlice | None) -> Tensor:
        if observation_slice is None:
            return torch.zeros((obs_batch.shape[0], 0), device=obs_batch.device, dtype=torch.long)
        return obs_batch[:, observation_slice.start : observation_slice.stop].to(dtype=torch.long)

    def _slot_component(self, stage_values: Tensor, offset: int) -> Tensor:
        if offset >= stage_values.shape[-1]:
            return torch.zeros(stage_values.shape[:2], device=stage_values.device, dtype=stage_values.dtype)
        return stage_values[..., offset]
