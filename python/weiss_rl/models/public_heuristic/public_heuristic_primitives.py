"""Primitive public-heuristic adapter methods for the structured policy head."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

import torch
from torch import Tensor

from weiss_rl.models.public_heuristic.public_heuristic_bias import combine_public_heuristic_scores
from weiss_rl.models.public_heuristic.public_heuristic_bias_mixin import StructuredPublicHeuristicBiasMixin
from weiss_rl.models.public_heuristic.public_heuristic_board_actions import (
    attack_public_heuristic_raw,
    move_public_heuristic_raw,
    play_public_heuristic_raw,
)
from weiss_rl.models.public_heuristic.public_heuristic_family_actions import (
    default_public_heuristic_raw,
    hand_public_heuristic_raw,
    index_public_heuristic_raw,
    slot_family_public_heuristic_raw,
)
from weiss_rl.models.public_heuristic.public_heuristic_slots import (
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    public_attack_profile,
    public_prefer_lower,
    public_slot_action_score,
    slot_preference_values,
)


class StructuredPublicHeuristicPrimitiveMixin(StructuredPublicHeuristicBiasMixin):
    """Small adapters from head state to pure public-heuristic scoring helpers."""

    def _slot_preference_values(self, slot_indices: Tensor, *, dtype: torch.dtype) -> Tensor:
        return slot_preference_values(slot_indices, self._public_slot_preference, dtype=dtype)

    def _public_prefer_lower(self, values: Tensor, *, dtype: torch.dtype) -> Tensor:
        return public_prefer_lower(values, dtype=dtype)

    def _public_slot_action_score(
        self,
        slot_values: Tensor,
        slot_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return public_slot_action_score(
            slot_values,
            slot_numeric,
            self._public_slot_preference,
            dtype=dtype,
        )

    def _combine_public_heuristic_scores(
        self,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return combine_public_heuristic_scores(score0, score1, score2, dtype=dtype)

    def _play_public_heuristic_raw(
        self,
        stage_slots: Tensor,
        target_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return play_public_heuristic_raw(
            stage_slots,
            target_numeric,
            self._public_slot_preference,
            dtype=dtype,
        )

    def _move_public_heuristic_raw(
        self,
        from_slots: Tensor,
        to_slots: Tensor,
        source_numeric: Tensor,
        target_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return move_public_heuristic_raw(
            from_slots,
            to_slots,
            source_numeric,
            target_numeric,
            self._public_slot_preference,
            dtype=dtype,
        )

    def _attack_public_heuristic_raw(
        self,
        slot_values: Tensor,
        attack_type_values: Tensor,
        source_numeric: Tensor,
        defender_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return attack_public_heuristic_raw(
            slot_values,
            attack_type_values,
            source_numeric,
            defender_numeric,
            self._public_slot_preference,
            direct_attack_type_id=self._direct_attack_type_id,
            frontal_attack_type_id=self._frontal_attack_type_id,
            side_attack_type_id=self._side_attack_type_id,
            dtype=dtype,
        )

    def _slot_family_public_heuristic_raw(
        self,
        family_ids: Tensor,
        slot_values: Tensor,
        slot_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return slot_family_public_heuristic_raw(
            family_ids,
            slot_values,
            slot_numeric,
            self._public_slot_preference,
            encore_pay_family_id=self._encore_pay_family_id,
            encore_decline_family_id=self._encore_decline_family_id,
            dtype=dtype,
        )

    def _public_attack_profile(
        self,
        self_stage_numeric: Tensor,
        opponent_stage_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        return public_attack_profile(
            self_stage_numeric,
            opponent_stage_numeric,
            front_row_count=len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS),
            dtype=dtype,
        )

    def _hand_public_heuristic_raw(
        self,
        family_ids: Tensor,
        hand_indices: Tensor,
        *,
        attackers_available: Tensor,
        front_defenders: Tensor,
        self_level_count: Tensor,
        self_clock_count: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        return hand_public_heuristic_raw(
            family_ids,
            hand_indices,
            attackers_available=attackers_available,
            front_defenders=front_defenders,
            self_level_count=self_level_count,
            self_clock_count=self_clock_count,
            climax_play_family_id=self._climax_play_family_id,
            clock_from_hand_family_id=self._clock_from_hand_family_id,
            main_event_family_id=self._main_event_family_id,
            mulligan_select_family_id=self._mulligan_select_family_id,
            dtype=dtype,
        )

    def _index_public_heuristic_raw(
        self,
        family_ids: Tensor,
        index_values: Tensor,
        *,
        choice_page_start: Tensor,
        choice_total: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        return index_public_heuristic_raw(
            family_ids,
            index_values,
            choice_page_start=choice_page_start,
            choice_total=choice_total,
            choice_select_family_id=self._choice_select_family_id,
            level_up_family_id=self._level_up_family_id,
            trigger_order_family_id=self._trigger_order_family_id,
            next_page_family_id=self._next_page_family_id,
            prev_page_family_id=self._prev_page_family_id,
            dtype=dtype,
        )

    def _default_public_heuristic_raw(
        self,
        family_ids: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return default_public_heuristic_raw(
            family_ids,
            mulligan_confirm_family_id=self._mulligan_confirm_family_id,
            pass_family_id=self._pass_family_id,
            dtype=dtype,
        )


__all__ = ["StructuredPublicHeuristicPrimitiveMixin"]
