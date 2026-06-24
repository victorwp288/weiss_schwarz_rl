"""Public-heuristic scoring blocks for non-attack candidate families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.models.public_heuristic.public_heuristic_hand_scoring import StructuredPublicHeuristicHandScoringMixin
from weiss_rl.models.public_heuristic.public_heuristic_slots import PUBLIC_HEURISTIC_CENTER_SLOT
from weiss_rl.public_heuristic.profiles import HeuristicPublicScoringProfile


class StructuredPublicHeuristicCandidateGroupMixin(StructuredPublicHeuristicHandScoringMixin):
    """Score public-heuristic priorities for each non-attack candidate family."""

    def _score_public_slot_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        self_stage_numeric: Tensor,
        row_indices_long: Tensor,
        slot_family_indices: Tensor,
        score0: Tensor,
        score1: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if slot_family_indices.numel() <= 0:
            return
        slot_rows = row_indices_long.index_select(0, slot_family_indices)
        slot_family_ids = scoring_plan.family_ids.index_select(0, slot_family_indices)
        slot_values = scoring_plan.arg0.index_select(0, slot_family_indices)
        _slot_context, slot_numeric = self._gather_stage_features_for_rows(
            observation_context["self_stage_context"],
            self_stage_numeric,
            slot_rows,
            slot_values,
        )
        del _slot_context
        slot_scores = self._public_slot_action_score(slot_values, slot_numeric, dtype=dtype)
        if self._encore_pay_family_id >= 0:
            encore_pay_mask = slot_family_ids == self._encore_pay_family_id
            if bool(encore_pay_mask.any().item()):
                pay_indices = slot_family_indices[encore_pay_mask]
                score0.index_fill_(0, pay_indices, float(scoring_profile.encore_pay_priority))
                score1.index_copy_(0, pay_indices, slot_scores[encore_pay_mask])
        if self._encore_decline_family_id >= 0:
            encore_decline_mask = slot_family_ids == self._encore_decline_family_id
            if bool(encore_decline_mask.any().item()):
                decline_indices = slot_family_indices[encore_decline_mask]
                score0.index_fill_(0, decline_indices, float(scoring_profile.encore_decline_priority))
                score1.index_copy_(0, decline_indices, slot_scores[encore_decline_mask])

    def _score_public_play_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        self_stage_numeric: Tensor,
        row_indices_long: Tensor,
        play_indices: Tensor,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if play_indices.numel() <= 0:
            return
        play_rows = row_indices_long.index_select(0, play_indices)
        play_hand_indices = scoring_plan.arg0.index_select(0, play_indices)
        play_stage_slots = scoring_plan.arg1.index_select(0, play_indices)
        _play_target_context, play_target_numeric = self._gather_stage_features_for_rows(
            observation_context["self_stage_context"],
            self_stage_numeric,
            play_rows,
            play_stage_slots,
        )
        del _play_target_context
        play_score = self._slot_preference_values(play_stage_slots, dtype=dtype)
        play_score = play_score + torch.where(
            play_stage_slots <= 2,
            play_stage_slots.new_full(play_stage_slots.shape, float(scoring_profile.play_front_bonus), dtype=dtype),
            torch.where(
                play_stage_slots <= 4,
                play_stage_slots.new_full(play_stage_slots.shape, float(scoring_profile.play_back_bonus), dtype=dtype),
                play_stage_slots.new_zeros(play_stage_slots.shape, dtype=dtype),
            ),
        )
        play_score = torch.where(
            play_target_numeric[:, 0].to(dtype=dtype) > 0.5,
            play_stage_slots.new_full(play_stage_slots.shape, -1000.0, dtype=dtype),
            play_score,
        )
        score0.index_fill_(0, play_indices, float(scoring_profile.play_priority))
        score1.index_copy_(0, play_indices, play_score)
        score2.index_copy_(0, play_indices, self._public_prefer_lower(play_hand_indices, dtype=dtype))

    def _score_public_index_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        index_family_indices: Tensor,
        row_indices_long: Tensor,
        choice_page_start: Tensor,
        choice_total: Tensor,
        score0: Tensor,
        score1: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if index_family_indices.numel() <= 0:
            return
        index_rows = row_indices_long.index_select(0, index_family_indices)
        index_family_ids = scoring_plan.family_ids.index_select(0, index_family_indices)
        index_values = scoring_plan.arg0.index_select(0, index_family_indices)
        self._score_public_choice_and_order_candidates(
            index_family_indices=index_family_indices,
            index_family_ids=index_family_ids,
            index_values=index_values,
            score0=score0,
            score1=score1,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_pager_candidates(
            index_family_indices=index_family_indices,
            index_family_ids=index_family_ids,
            index_rows=index_rows,
            choice_page_start=choice_page_start,
            choice_total=choice_total,
            score0=score0,
            score1=score1,
            scoring_profile=scoring_profile,
        )

    def _score_public_choice_and_order_candidates(
        self: Any,
        *,
        index_family_indices: Tensor,
        index_family_ids: Tensor,
        index_values: Tensor,
        score0: Tensor,
        score1: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        family_rules = (
            (self._choice_select_family_id, scoring_profile.choice_select_priority),
            (self._level_up_family_id, scoring_profile.level_up_priority),
            (self._trigger_order_family_id, scoring_profile.trigger_order_priority),
        )
        for family_id, priority in family_rules:
            if family_id < 0:
                continue
            family_mask = index_family_ids == family_id
            if bool(family_mask.any().item()):
                family_indices = index_family_indices[family_mask]
                score0.index_fill_(0, family_indices, float(priority))
                score1.index_copy_(0, family_indices, self._public_prefer_lower(index_values[family_mask], dtype=dtype))

    def _score_public_pager_candidates(
        self: Any,
        *,
        index_family_indices: Tensor,
        index_family_ids: Tensor,
        index_rows: Tensor,
        choice_page_start: Tensor,
        choice_total: Tensor,
        score0: Tensor,
        score1: Tensor,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if self._next_page_family_id >= 0:
            next_mask = index_family_ids == self._next_page_family_id
            if bool(next_mask.any().item()):
                next_indices = index_family_indices[next_mask]
                next_rows = index_rows[next_mask]
                score0.index_fill_(0, next_indices, float(scoring_profile.pager_priority))
                score1.index_copy_(
                    0,
                    next_indices,
                    torch.clamp(
                        choice_total.index_select(0, next_rows) - (choice_page_start.index_select(0, next_rows) + 16.0),
                        min=0.0,
                    ),
                )
        if self._prev_page_family_id >= 0:
            prev_mask = index_family_ids == self._prev_page_family_id
            if bool(prev_mask.any().item()):
                prev_indices = index_family_indices[prev_mask]
                prev_rows = index_rows[prev_mask]
                score0.index_fill_(0, prev_indices, float(scoring_profile.pager_priority))
                score1.index_copy_(0, prev_indices, torch.clamp(choice_page_start.index_select(0, prev_rows), min=0.0))

    def _score_public_move_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        self_stage_numeric: Tensor,
        row_indices_long: Tensor,
        move_indices: Tensor,
        score0: Tensor,
        score1: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if move_indices.numel() <= 0:
            return
        move_rows = row_indices_long.index_select(0, move_indices)
        move_from_slots = scoring_plan.arg0.index_select(0, move_indices)
        move_to_slots = scoring_plan.arg1.index_select(0, move_indices)
        _move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
            observation_context["self_stage_context"],
            self_stage_numeric,
            move_rows,
            move_from_slots,
        )
        del _move_source_context
        _move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
            observation_context["self_stage_context"],
            self_stage_numeric,
            move_rows,
            move_to_slots,
        )
        del _move_target_context
        source_pref = self._slot_preference_values(move_from_slots, dtype=dtype)
        target_pref = self._slot_preference_values(move_to_slots, dtype=dtype)
        bonus = torch.zeros(move_to_slots.shape, dtype=dtype, device=row_indices_long.device)
        bonus = bonus + (
            ((move_from_slots >= 3) & (move_to_slots <= 2)).to(dtype=dtype)
            * float(scoring_profile.move_back_to_front_bonus)
        )
        bonus = bonus + (
            ((move_to_slots == PUBLIC_HEURISTIC_CENTER_SLOT) & (move_from_slots != PUBLIC_HEURISTIC_CENTER_SLOT)).to(
                dtype=dtype
            )
            * float(scoring_profile.move_center_bonus)
        )
        legal = (move_source_numeric[:, 0].to(dtype=dtype) > 0.5) & (move_target_numeric[:, 0].to(dtype=dtype) <= 0.5)
        move_score = torch.where(
            legal,
            (target_pref - source_pref) + bonus,
            move_to_slots.new_full(move_to_slots.shape, -1000.0, dtype=dtype),
        )
        score0.index_fill_(0, move_indices, float(scoring_profile.move_priority))
        score1.index_copy_(0, move_indices, move_score)

    def _score_public_default_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        default_indices: Tensor,
        score0: Tensor,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if default_indices.numel() <= 0:
            return
        default_family_ids = scoring_plan.family_ids.index_select(0, default_indices)
        if self._mulligan_confirm_family_id >= 0:
            mulligan_confirm_mask = default_family_ids == self._mulligan_confirm_family_id
            if bool(mulligan_confirm_mask.any().item()):
                score0.index_fill_(
                    0,
                    default_indices[mulligan_confirm_mask],
                    float(scoring_profile.mulligan_confirm_priority),
                )
        if self._pass_family_id >= 0:
            pass_mask = default_family_ids == self._pass_family_id
            if bool(pass_mask.any().item()):
                score0.index_fill_(0, default_indices[pass_mask], float(scoring_profile.pass_priority))


__all__ = ["StructuredPublicHeuristicCandidateGroupMixin"]
