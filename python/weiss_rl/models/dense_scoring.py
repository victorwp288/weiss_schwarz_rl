"""Dense candidate scoring mixin for the structured action head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.tensor_ops import optional_embedding

_optional_embedding = optional_embedding


class StructuredDenseScoringMixin:
    """Dense candidate scoring helpers used by `_StructuredLegalActionHead`."""

    def _score_candidates(
        self: Any,
        state_repr: Tensor,
        row_indices: Tensor,
        candidate_ids: Tensor,
        observation_context: Mapping[str, Tensor],
        candidate_meta: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        row_indices_long = row_indices.to(dtype=torch.long)
        row_states = state_repr.index_select(0, row_indices_long)
        hand_indices: Tensor | None = None
        stage_slots: Tensor | None = None
        from_slots: Tensor | None = None
        to_slots: Tensor | None = None
        attack_slots: Tensor | None = None
        attack_types: Tensor | None = None
        generic_indices: Tensor | None = None
        meta_arg0: Tensor | None = None
        meta_arg1: Tensor | None = None
        if candidate_meta is None:
            (
                family_ids,
                hand_indices,
                stage_slots,
                from_slots,
                to_slots,
                attack_slots,
                attack_types,
                generic_indices,
            ) = self._resolve_candidate_components(candidate_ids, None)
        else:
            family_ids = candidate_meta[:, 0].to(dtype=torch.long)
            meta_arg0 = candidate_meta[:, 1].to(dtype=torch.long)
            meta_arg1 = candidate_meta[:, 2].to(dtype=torch.long)
            meta_arg0 = torch.where(meta_arg0 == self._meta_unused, torch.full_like(meta_arg0, -1), meta_arg0)
            meta_arg1 = torch.where(meta_arg1 == self._meta_unused, torch.full_like(meta_arg1, -1), meta_arg1)

        def component_from_meta_or_catalog(
            meta_values: Tensor | None, catalog_values: Tensor | None, mask: Tensor
        ) -> Tensor:
            if meta_values is not None:
                return meta_values[mask]
            if catalog_values is None:
                raise RuntimeError("candidate catalog metadata is required when packed metadata is absent")
            return catalog_values[mask]

        family_embeddings = self.family_embedding(family_ids).to(dtype=row_states.dtype)
        scores = row_states.new_empty((candidate_ids.shape[0],), dtype=row_states.dtype)
        public_bias_scale = self._public_heuristic_logit_bias_scale_for(scoring_mode)
        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]

        play_mask = family_ids == self._play_character_family_id
        if torch.any(play_mask):
            play_rows = row_indices_long[play_mask]
            play_row_states = row_states[play_mask]
            play_hand_indices = component_from_meta_or_catalog(meta_arg0, hand_indices, play_mask)
            play_stage_slots = component_from_meta_or_catalog(meta_arg1, stage_slots, play_mask)
            play_hand_present, play_hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                play_rows,
                play_hand_indices,
                dtype=row_states.dtype,
            )
            play_target_context, play_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                play_rows,
                play_stage_slots,
            )
            play_scores = self._score_candidate_group(
                play_row_states,
                feature_sections=(
                    (family_embeddings[play_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (play_hand_card_embeddings, (self._hand_card_feature_offset, self._stage_slot_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, play_stage_slots).to(dtype=row_states.dtype),
                        (self._stage_slot_feature_offset, self._from_slot_feature_offset),
                    ),
                    (
                        play_target_context.to(dtype=row_states.dtype),
                        (self._play_target_context_offset, self._move_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (play_hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),
                    ((1.0 - play_target_numeric[:, :1]).to(dtype=row_states.dtype), (8,)),
                ),
                constant_numeric_ones=(1, 9),
            )
            if public_bias_scale > 0.0:
                play_scores = self._apply_public_heuristic_bias(
                    play_scores,
                    self._play_public_heuristic_raw(
                        play_stage_slots,
                        play_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[play_mask],
                )
            scores[play_mask] = play_scores

        hand_family_ids = (
            self._main_event_family_id,
            self._clock_from_hand_family_id,
            self._climax_play_family_id,
            self._mulligan_select_family_id,
        )
        hand_mask = torch.zeros_like(play_mask)
        for family_id in hand_family_ids:
            if family_id >= 0:
                hand_mask |= family_ids == family_id
        if torch.any(hand_mask):
            hand_rows = row_indices_long[hand_mask]
            hand_row_states = row_states[hand_mask]
            hand_family_indices = component_from_meta_or_catalog(meta_arg0, hand_indices, hand_mask)
            hand_present, hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                hand_rows,
                hand_family_indices,
                dtype=row_states.dtype,
            )
            scores[hand_mask] = self._score_candidate_group(
                hand_row_states,
                feature_sections=(
                    (family_embeddings[hand_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (hand_card_embeddings, (self._hand_card_feature_offset, self._stage_slot_feature_offset)),
                ),
                numeric_sections=((hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),),
                constant_numeric_ones=(8, 9),
            )
            if public_bias_scale > 0.0:
                attackers_available, front_defenders = self._public_attack_profile(
                    self_stage_numeric,
                    opponent_stage_numeric,
                    dtype=row_states.dtype,
                )
                scores[hand_mask] = self._apply_public_heuristic_bias(
                    scores[hand_mask],
                    self._hand_public_heuristic_raw(
                        family_ids[hand_mask],
                        hand_family_indices,
                        attackers_available=attackers_available.index_select(0, hand_rows),
                        front_defenders=front_defenders.index_select(0, hand_rows),
                        self_level_count=observation_context["self_level_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        self_clock_count=observation_context["self_clock_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[hand_mask],
                )

        move_mask = family_ids == self._main_move_family_id
        if torch.any(move_mask):
            move_rows = row_indices_long[move_mask]
            move_row_states = row_states[move_mask]
            move_from_slots = component_from_meta_or_catalog(meta_arg0, from_slots, move_mask)
            move_to_slots = component_from_meta_or_catalog(meta_arg1, to_slots, move_mask)
            move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_from_slots,
            )
            move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_to_slots,
            )
            move_scores = self._score_candidate_group(
                move_row_states,
                feature_sections=(
                    (family_embeddings[move_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, move_from_slots).to(dtype=row_states.dtype),
                        (self._from_slot_feature_offset, self._to_slot_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, move_to_slots).to(dtype=row_states.dtype),
                        (self._to_slot_feature_offset, self._attack_slot_feature_offset),
                    ),
                    (
                        move_source_context.to(dtype=row_states.dtype),
                        (self._move_source_context_offset, self._move_target_context_offset),
                    ),
                    (
                        move_target_context.to(dtype=row_states.dtype),
                        (self._move_target_context_offset, self._attack_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (move_source_numeric[:, :1].to(dtype=row_states.dtype), (7,)),
                    ((1.0 - move_target_numeric[:, :1]).to(dtype=row_states.dtype), (9,)),
                ),
                constant_numeric_ones=(2, 3, 8),
            )
            if public_bias_scale > 0.0:
                move_scores = self._apply_public_heuristic_bias(
                    move_scores,
                    self._move_public_heuristic_raw(
                        move_from_slots,
                        move_to_slots,
                        move_source_numeric,
                        move_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[move_mask],
                )
            scores[move_mask] = move_scores

        attack_mask = family_ids == self._attack_family_id
        if torch.any(attack_mask):
            attack_rows = row_indices_long[attack_mask]
            attack_row_states = row_states[attack_mask]
            attack_slot_values = component_from_meta_or_catalog(meta_arg0, attack_slots, attack_mask)
            attack_type_values = component_from_meta_or_catalog(meta_arg1, attack_types, attack_mask)
            attack_source_context, attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                opponent_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            attack_scores = self._score_candidate_group(
                attack_row_states,
                feature_sections=(
                    (family_embeddings[attack_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, attack_slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        _optional_embedding(self.attack_type_embedding, attack_type_values).to(dtype=row_states.dtype),
                        (self._attack_type_feature_offset, self._play_target_context_offset),
                    ),
                    (
                        attack_source_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                    (
                        defender_context.to(dtype=row_states.dtype),
                        (self._defender_context_offset, self._numeric_feature_offset),
                    ),
                ),
                numeric_sections=((defender_numeric[:, :1].to(dtype=row_states.dtype), (10,)),),
                constant_numeric_ones=(4, 5, 8, 9),
            )
            if public_bias_scale > 0.0:
                attack_scores = self._apply_public_heuristic_bias(
                    attack_scores,
                    self._attack_public_heuristic_raw(
                        attack_slot_values,
                        attack_type_values,
                        attack_source_numeric,
                        defender_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[attack_mask],
                )
            scores[attack_mask] = attack_scores

        slot_mask = torch.zeros_like(play_mask)
        for family_id in self._slot_family_ids:
            slot_mask |= family_ids == family_id
        if torch.any(slot_mask):
            slot_rows = row_indices_long[slot_mask]
            slot_row_states = row_states[slot_mask]
            slot_values = component_from_meta_or_catalog(meta_arg0, attack_slots, slot_mask)
            slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            slot_scores = self._score_candidate_group(
                slot_row_states,
                feature_sections=(
                    (family_embeddings[slot_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        slot_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                ),
                numeric_sections=((slot_numeric[:, :1].to(dtype=row_states.dtype), (7,)),),
            )
            if public_bias_scale > 0.0:
                slot_scores = self._apply_public_heuristic_bias(
                    slot_scores,
                    self._slot_family_public_heuristic_raw(
                        family_ids[slot_mask],
                        slot_values,
                        slot_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[slot_mask],
                )
            scores[slot_mask] = slot_scores

        index_mask = torch.zeros_like(play_mask)
        for family_id in self._index_family_ids:
            index_mask |= family_ids == family_id
        if torch.any(index_mask):
            index_rows = row_indices_long[index_mask]
            index_row_states = row_states[index_mask]
            index_values = component_from_meta_or_catalog(meta_arg0, generic_indices, index_mask)
            scores[index_mask] = self._score_candidate_group(
                index_row_states,
                feature_sections=(
                    (family_embeddings[index_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        self._project_generic_index_features(index_values, dtype=row_states.dtype),
                        (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                    ),
                ),
                numeric_sections=((torch.clamp(index_values.to(dtype=row_states.dtype), min=0.0).unsqueeze(1), (6,)),),
            )
            if public_bias_scale > 0.0:
                scores[index_mask] = self._apply_public_heuristic_bias(
                    scores[index_mask],
                    self._index_public_heuristic_raw(
                        family_ids[index_mask],
                        index_values,
                        choice_page_start=observation_context["choice_page_start"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        choice_total=observation_context["choice_total"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[index_mask],
                )

        default_mask = ~(play_mask | hand_mask | move_mask | attack_mask | slot_mask | index_mask)
        if torch.any(default_mask):
            default_row_states = row_states[default_mask]
            default_generic_indices = component_from_meta_or_catalog(meta_arg0, generic_indices, default_mask)
            default_scores = self._score_candidate_group(
                default_row_states,
                feature_sections=(
                    (family_embeddings[default_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                ),
                numeric_sections=(((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),),
                constant_numeric_ones=(8, 9),
            )
            if public_bias_scale > 0.0:
                default_scores = self._apply_public_heuristic_bias(
                    default_scores,
                    self._default_public_heuristic_raw(
                        family_ids[default_mask],
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[default_mask],
                )
            scores[default_mask] = default_scores

        return scores + self.family_bias.index_select(0, family_ids).to(dtype=row_states.dtype)
