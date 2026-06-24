"""Public-heuristic scoring blocks for hand-origin candidate families."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.public_heuristic.profiles import HeuristicPublicScoringProfile


class StructuredPublicHeuristicHandScoringMixin:
    """Score public-heuristic priorities for hand-origin candidate families."""

    def _score_public_hand_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        hand_indices: Tensor,
        row_indices_long: Tensor,
        attackers_available: Tensor,
        front_defenders: Tensor,
        self_level_count: Tensor,
        self_clock_count: Tensor,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if hand_indices.numel() <= 0:
            return
        hand_rows = row_indices_long.index_select(0, hand_indices)
        hand_family_ids = scoring_plan.family_ids.index_select(0, hand_indices)
        hand_indices_values = scoring_plan.arg0.index_select(0, hand_indices)
        self._score_public_climax_candidates(
            hand_indices=hand_indices,
            hand_rows=hand_rows,
            hand_family_ids=hand_family_ids,
            hand_indices_values=hand_indices_values,
            attackers_available=attackers_available,
            front_defenders=front_defenders,
            score0=score0,
            score1=score1,
            score2=score2,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_clock_candidates(
            hand_indices=hand_indices,
            hand_rows=hand_rows,
            hand_family_ids=hand_family_ids,
            hand_indices_values=hand_indices_values,
            self_level_count=self_level_count,
            self_clock_count=self_clock_count,
            score0=score0,
            score1=score1,
            score2=score2,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_event_and_mulligan_candidates(
            hand_indices=hand_indices,
            hand_family_ids=hand_family_ids,
            hand_indices_values=hand_indices_values,
            score0=score0,
            score1=score1,
            score2=score2,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )

    def _score_public_climax_candidates(
        self: Any,
        *,
        hand_indices: Tensor,
        hand_rows: Tensor,
        hand_family_ids: Tensor,
        hand_indices_values: Tensor,
        attackers_available: Tensor,
        front_defenders: Tensor,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if self._climax_play_family_id < 0:
            return
        climax_mask = hand_family_ids == self._climax_play_family_id
        if not bool(climax_mask.any().item()):
            return
        climax_indices = hand_indices[climax_mask]
        climax_rows = hand_rows[climax_mask]
        score0.index_fill_(0, climax_indices, float(scoring_profile.climax_priority))
        score1.index_copy_(
            0,
            climax_indices,
            attackers_available.index_select(0, climax_rows) * float(scoring_profile.climax_attacker_scale)
            + front_defenders.index_select(0, climax_rows) * float(scoring_profile.climax_defender_scale)
            + torch.where(
                attackers_available.index_select(0, climax_rows) > 0.0,
                hand_indices_values.new_full(
                    climax_rows.shape,
                    float(scoring_profile.climax_active_bonus),
                    dtype=dtype,
                ),
                hand_indices_values.new_full(
                    climax_rows.shape,
                    float(scoring_profile.climax_inactive_bonus),
                    dtype=dtype,
                ),
            ),
        )
        score2.index_copy_(0, climax_indices, self._public_prefer_lower(hand_indices_values[climax_mask], dtype=dtype))

    def _score_public_clock_candidates(
        self: Any,
        *,
        hand_indices: Tensor,
        hand_rows: Tensor,
        hand_family_ids: Tensor,
        hand_indices_values: Tensor,
        self_level_count: Tensor,
        self_clock_count: Tensor,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if self._clock_from_hand_family_id < 0:
            return
        clock_mask = hand_family_ids == self._clock_from_hand_family_id
        if not bool(clock_mask.any().item()):
            return
        clock_indices = hand_indices[clock_mask]
        clock_rows = hand_rows[clock_mask]
        level_counts = self_level_count.index_select(0, clock_rows)
        clock_counts = self_clock_count.index_select(0, clock_rows)
        score0.index_fill_(0, clock_indices, float(scoring_profile.clock_priority))
        score1.index_copy_(
            0,
            clock_indices,
            torch.where(
                (level_counts <= 0.0) & (clock_counts < 6.0),
                float(scoring_profile.early_clock_score) - clock_counts,
                clock_counts.new_full(clock_counts.shape, float(scoring_profile.late_clock_score), dtype=dtype),
            ),
        )
        score2.index_copy_(0, clock_indices, self._public_prefer_lower(hand_indices_values[clock_mask], dtype=dtype))

    def _score_public_event_and_mulligan_candidates(
        self: Any,
        *,
        hand_indices: Tensor,
        hand_family_ids: Tensor,
        hand_indices_values: Tensor,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if self._main_event_family_id >= 0:
            event_mask = hand_family_ids == self._main_event_family_id
            if bool(event_mask.any().item()):
                event_indices = hand_indices[event_mask]
                score0.index_fill_(0, event_indices, float(scoring_profile.event_priority))
                score1.index_fill_(0, event_indices, 10.0)
                score2.index_copy_(
                    0,
                    event_indices,
                    self._public_prefer_lower(hand_indices_values[event_mask], dtype=dtype),
                )
        if self._mulligan_select_family_id >= 0:
            mulligan_mask = hand_family_ids == self._mulligan_select_family_id
            if bool(mulligan_mask.any().item()):
                mulligan_indices = hand_indices[mulligan_mask]
                score0.index_fill_(0, mulligan_indices, float(scoring_profile.mulligan_select_priority))
                score1.index_copy_(
                    0,
                    mulligan_indices,
                    self._public_prefer_lower(hand_indices_values[mulligan_mask], dtype=dtype),
                )


__all__ = ["StructuredPublicHeuristicHandScoringMixin"]
