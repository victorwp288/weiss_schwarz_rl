"""Packed candidate scoring mixin for the structured action head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.action_plans import PackedScoringPlan
from weiss_rl.models.candidate_partitioning import partition_candidate_family_indices
from weiss_rl.models.tensor_ops import optional_embedding, packed_row_indices

_PackedScoringPlan = PackedScoringPlan
_optional_embedding = optional_embedding
_packed_row_indices = packed_row_indices


class StructuredPackedScoringMixin:
    """Packed candidate scoring helpers used by `_StructuredLegalActionHead`."""

    def _build_packed_scoring_plan(
        self: Any,
        *,
        candidate_ids: Tensor,
        offsets: Tensor,
        candidate_meta: Tensor | None,
    ) -> _PackedScoringPlan:
        if candidate_meta is None:
            family_ids = self._family_ids.index_select(0, candidate_ids)
            arg0 = self._action_arg0.index_select(0, candidate_ids)
            arg1 = self._action_arg1.index_select(0, candidate_ids)
        else:
            family_ids = candidate_meta[:, 0].to(dtype=torch.long)
            arg0 = candidate_meta[:, 1].to(dtype=torch.long)
            arg1 = candidate_meta[:, 2].to(dtype=torch.long)
            meta_unused = torch.full_like(arg0, self._meta_unused)
            arg0 = torch.where(arg0 == meta_unused, torch.full_like(arg0, -1), arg0)
            arg1 = torch.where(arg1 == meta_unused, torch.full_like(arg1, -1), arg1)
        return _PackedScoringPlan(
            row_indices=_packed_row_indices(offsets),
            family_ids=family_ids,
            arg0=arg0,
            arg1=arg1,
        )

    def _partition_candidate_family_indices(
        self: Any,
        family_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return partition_candidate_family_indices(
            family_ids,
            play_character_family_id=int(self._play_character_family_id),
            hand_family_ids=tuple(int(family_id) for family_id in self._hand_family_ids),
            main_move_family_id=int(self._main_move_family_id),
            attack_family_id=int(self._attack_family_id),
            slot_family_ids=tuple(int(family_id) for family_id in self._slot_family_ids),
            index_family_ids=tuple(int(family_id) for family_id in self._index_family_ids),
        )

    def _project_generic_index_features(
        self: Any,
        index_values: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        valid = index_values >= 0
        embedded = _optional_embedding(self.generic_index_embedding, index_values).to(dtype=dtype)
        projected = self.generic_candidate_projection(embedded)
        return projected * valid.unsqueeze(1).to(dtype=dtype)

    def _score_candidates_chunked(
        self: Any,
        state_repr: Tensor,
        row_indices: Tensor,
        candidate_ids: Tensor,
        observation_context: Mapping[str, Tensor],
        *,
        candidate_meta: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if candidate_ids.numel() == 0:
            return state_repr.new_zeros((0,))
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "learner" and state_repr.device.type == "cuda":
            chunk_size = max(chunk_size, int(self._cuda_learner_candidate_scoring_chunk_size))
        for start in range(0, int(candidate_ids.numel()), chunk_size):
            end = min(start + chunk_size, int(candidate_ids.numel()))
            scores_chunks.append(
                self._score_candidates(
                    state_repr,
                    row_indices[start:end],
                    candidate_ids[start:end],
                    observation_context,
                    candidate_meta=None if candidate_meta is None else candidate_meta[start:end],
                    scoring_mode=resolved_mode,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _score_packed_candidates_chunked(
        self: Any,
        state_repr: Tensor,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if scoring_plan.candidate_count == 0:
            return state_repr.new_zeros((0,))
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "learner" and state_repr.device.type == "cuda":
            chunk_size = max(chunk_size, int(self._cuda_learner_candidate_scoring_chunk_size))
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            scores_chunks.append(
                self._score_packed_candidates_plan(
                    state_repr,
                    scoring_plan.slice(start, end),
                    observation_context,
                    scoring_mode=resolved_mode,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _project_packed_candidate_representations(
        self: Any,
        state_repr: Tensor,
        legal_actions: Any,
        observation_context: Mapping[str, Tensor],
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("projecting packed candidate representations requires packed ids and offsets")
        ids = torch.as_tensor(legal_actions.ids, device=state_repr.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=state_repr.device, dtype=torch.long)
        meta = (
            None
            if legal_actions.meta is None
            else torch.as_tensor(legal_actions.meta, device=state_repr.device, dtype=torch.long)
        )
        if offsets.ndim != 1 or offsets.numel() != state_repr.shape[0] + 1:
            raise ValueError(f"packed legal offsets must have shape ({state_repr.shape[0] + 1},)")
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(ids.numel()):
            raise ValueError("packed legal offsets must be a valid prefix sum")
        if ids.numel() == 0:
            return state_repr.new_zeros((0, state_repr.shape[1]))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=ids,
            offsets=offsets,
            candidate_meta=meta,
        )
        repr_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "learner" and state_repr.device.type == "cuda":
            chunk_size = max(chunk_size, int(self._cuda_learner_candidate_scoring_chunk_size))
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            repr_chunks.append(
                self._project_packed_candidates_plan(
                    state_repr,
                    scoring_plan.slice(start, end),
                    observation_context,
                    scoring_mode=resolved_mode,
                )
            )
        return torch.cat(repr_chunks, dim=0)

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

    def _score_packed_candidates_plan(
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
        scores = row_states.new_empty((scoring_plan.candidate_count,), dtype=row_states.dtype)
        public_bias_scale = self._public_heuristic_logit_bias_scale_for(scoring_mode)
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
            play_row_states = row_states.index_select(0, play_indices)
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
            play_scores = self._score_candidate_group(
                play_row_states,
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
                    family_ids=scoring_plan.family_ids.index_select(0, play_indices),
                )
            scores.index_copy_(
                0,
                play_indices,
                play_scores,
            )

        if hand_indices.numel() > 0:
            hand_rows = row_indices_long.index_select(0, hand_indices)
            hand_row_states = row_states.index_select(0, hand_indices)
            hand_family_indices = scoring_plan.arg0.index_select(0, hand_indices)
            hand_present, hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                hand_rows,
                hand_family_indices,
                dtype=row_states.dtype,
            )
            hand_scores = self._score_candidate_group(
                hand_row_states,
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
            )
            if public_bias_scale > 0.0:
                attackers_available, front_defenders = self._public_attack_profile(
                    self_stage_numeric,
                    opponent_stage_numeric,
                    dtype=row_states.dtype,
                )
                hand_scores = self._apply_public_heuristic_bias(
                    hand_scores,
                    self._hand_public_heuristic_raw(
                        scoring_plan.family_ids.index_select(0, hand_indices),
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
                    family_ids=scoring_plan.family_ids.index_select(0, hand_indices),
                )
            scores.index_copy_(0, hand_indices, hand_scores)

        if move_indices.numel() > 0:
            move_rows = row_indices_long.index_select(0, move_indices)
            move_row_states = row_states.index_select(0, move_indices)
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
            move_scores = self._score_candidate_group(
                move_row_states,
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
                    family_ids=scoring_plan.family_ids.index_select(0, move_indices),
                )
            scores.index_copy_(
                0,
                move_indices,
                move_scores,
            )

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_row_states = row_states.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
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
                    (
                        family_embeddings.index_select(0, attack_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
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
                scoring_mode=scoring_mode,
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
                    family_ids=scoring_plan.family_ids.index_select(0, attack_indices),
                )
            scores.index_copy_(
                0,
                attack_indices,
                attack_scores,
            )

        if slot_family_indices.numel() > 0:
            slot_rows = row_indices_long.index_select(0, slot_family_indices)
            slot_row_states = row_states.index_select(0, slot_family_indices)
            slot_family_ids = scoring_plan.family_ids.index_select(0, slot_family_indices)
            slot_values = scoring_plan.arg0.index_select(0, slot_family_indices)
            slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            slot_scores = self._score_candidate_group(
                slot_row_states,
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
            )
            if public_bias_scale > 0.0:
                slot_scores = self._apply_public_heuristic_bias(
                    slot_scores,
                    self._slot_family_public_heuristic_raw(
                        slot_family_ids,
                        slot_values,
                        slot_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=slot_family_ids,
                )
            scores.index_copy_(0, slot_family_indices, slot_scores)

        if index_family_indices.numel() > 0:
            index_rows = row_indices_long.index_select(0, index_family_indices)
            index_row_states = row_states.index_select(0, index_family_indices)
            index_values = scoring_plan.arg0.index_select(0, index_family_indices)
            index_scores = self._score_candidate_group(
                index_row_states,
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
                numeric_sections=((torch.clamp(index_values.to(dtype=row_states.dtype), min=0.0).unsqueeze(1), (6,)),),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                index_scores = self._apply_public_heuristic_bias(
                    index_scores,
                    self._index_public_heuristic_raw(
                        scoring_plan.family_ids.index_select(0, index_family_indices),
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
                    family_ids=scoring_plan.family_ids.index_select(0, index_family_indices),
                )
            scores.index_copy_(0, index_family_indices, index_scores)

        if default_indices.numel() > 0:
            default_row_states = row_states.index_select(0, default_indices)
            default_generic_indices = scoring_plan.arg0.index_select(0, default_indices)
            default_scores = self._score_candidate_group(
                default_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, default_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                ),
                numeric_sections=(((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),),
                constant_numeric_ones=(8, 9),
                scoring_mode=scoring_mode,
            )
            default_family_ids = scoring_plan.family_ids.index_select(0, default_indices)
            if public_bias_scale > 0.0:
                default_scores = self._apply_public_heuristic_bias(
                    default_scores,
                    self._default_public_heuristic_raw(
                        default_family_ids,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=default_family_ids,
                )
            scores.index_copy_(
                0,
                default_indices,
                default_scores,
            )

        return scores + self.family_bias.index_select(0, scoring_plan.family_ids).to(dtype=row_states.dtype)
