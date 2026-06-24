"""Packed legal-candidate representation projection for the structured head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.models.backbone.tensor_ops import optional_embedding

_PackedScoringPlan = PackedScoringPlan
_optional_embedding = optional_embedding


class StructuredPackedProjectionMixin:
    """Builds per-candidate representations before packed logits are scored."""

    def _project_packed_candidates_plan(
        self: Any,
        state_repr: Tensor,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        row_indices_long = scoring_plan.row_indices.to(dtype=torch.long)
        row_states = state_repr.index_select(0, row_indices_long)
        family_embeddings = self.family_embedding(scoring_plan.family_ids).to(dtype=row_states.dtype)
        candidate_repr = row_states.new_empty(
            (scoring_plan.candidate_count, row_states.shape[1]), dtype=row_states.dtype
        )
        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]
        (
            play_indices,
            hand_indices,
            move_indices,
            attack_indices,
            slot_family_indices,
            index_family_indices,
            default_indices,
        ) = self._partition_candidate_family_indices(scoring_plan.family_ids)

        if play_indices.numel() > 0:
            play_rows = row_indices_long.index_select(0, play_indices)
            play_hand_indices = scoring_plan.arg0.index_select(0, play_indices)
            play_stage_slots = scoring_plan.arg1.index_select(0, play_indices)
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
            candidate_repr.index_copy_(
                0,
                play_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, play_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                        (
                            play_hand_card_embeddings,
                            (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                        ),
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
                    scoring_mode=scoring_mode,
                ),
            )

        if hand_indices.numel() > 0:
            hand_rows = row_indices_long.index_select(0, hand_indices)
            hand_family_indices = scoring_plan.arg0.index_select(0, hand_indices)
            hand_present, hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                hand_rows,
                hand_family_indices,
                dtype=row_states.dtype,
            )
            candidate_repr.index_copy_(
                0,
                hand_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, hand_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                        (
                            hand_card_embeddings,
                            (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                        ),
                    ),
                    numeric_sections=((hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),),
                    constant_numeric_ones=(8, 9),
                    scoring_mode=scoring_mode,
                ),
            )

        if move_indices.numel() > 0:
            move_rows = row_indices_long.index_select(0, move_indices)
            move_from_slots = scoring_plan.arg0.index_select(0, move_indices)
            move_to_slots = scoring_plan.arg1.index_select(0, move_indices)
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
            candidate_repr.index_copy_(
                0,
                move_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, move_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
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
                    scoring_mode=scoring_mode,
                ),
            )

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
            attack_source_context, _attack_source_numeric = self._gather_stage_features_for_rows(
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
            candidate_repr.index_copy_(
                0,
                attack_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, attack_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                        (
                            _optional_embedding(self.slot_embedding, attack_slot_values).to(dtype=row_states.dtype),
                            (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                        ),
                        (
                            _optional_embedding(self.attack_type_embedding, attack_type_values).to(
                                dtype=row_states.dtype
                            ),
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
                    scoring_mode=scoring_mode,
                ),
            )

        if slot_family_indices.numel() > 0:
            slot_rows = row_indices_long.index_select(0, slot_family_indices)
            slot_values = scoring_plan.arg0.index_select(0, slot_family_indices)
            slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            candidate_repr.index_copy_(
                0,
                slot_family_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, slot_family_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
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
                    scoring_mode=scoring_mode,
                ),
            )

        if index_family_indices.numel() > 0:
            index_values = scoring_plan.arg0.index_select(0, index_family_indices)
            candidate_repr.index_copy_(
                0,
                index_family_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, index_family_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                        (
                            self._project_generic_index_features(index_values, dtype=row_states.dtype),
                            (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                        ),
                    ),
                    numeric_sections=(
                        (torch.clamp(index_values.to(dtype=row_states.dtype), min=0.0).unsqueeze(1), (6,)),
                    ),
                    scoring_mode=scoring_mode,
                ),
            )

        if default_indices.numel() > 0:
            default_generic_indices = scoring_plan.arg0.index_select(0, default_indices)
            candidate_repr.index_copy_(
                0,
                default_indices,
                self._project_candidate_sections(
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, default_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                    ),
                    numeric_sections=(((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),),
                    constant_numeric_ones=(8, 9),
                    scoring_mode=scoring_mode,
                ),
            )

        return candidate_repr


__all__ = ["StructuredPackedProjectionMixin"]
