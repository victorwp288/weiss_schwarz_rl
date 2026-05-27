"""Structured policy-head public heuristic scoring mixin."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from weiss_rl.eval.heuristic_public import HeuristicPublicScoringProfile
from weiss_rl.models.action_plans import PackedScoringPlan
from weiss_rl.models.public_heuristics import (
    PUBLIC_HEURISTIC_CENTER_SLOT,
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    apply_public_heuristic_bias,
    attack_public_heuristic_raw,
    combine_public_heuristic_scores,
    default_public_heuristic_raw,
    hand_public_heuristic_raw,
    index_public_heuristic_raw,
    move_public_heuristic_raw,
    play_public_heuristic_raw,
    public_attack_profile,
    public_prefer_lower,
    public_slot_action_score,
    slot_family_public_heuristic_raw,
    slot_preference_values,
)


class StructuredPublicHeuristicScoringMixin:
    """Adapter methods for public-heuristic structured action scoring.

    The owning policy head provides the tensors, embeddings, and candidate
    partition helpers. Keeping these methods in a mixin preserves the old
    private method surface while moving the heuristic-specific scoring rules
    out of the core model file.
    """

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

    def _public_heuristic_logit_bias_scale_for(self, scoring_mode: str) -> float:
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "actor":
            return float(self._public_heuristic_actor_logit_bias_scale)
        return float(self._public_heuristic_logit_bias_scale)

    def _apply_public_heuristic_bias(
        self,
        scores: Tensor,
        raw_scores: Tensor,
        *,
        scale: float,
        family_ids: Tensor | None = None,
    ) -> Tensor:
        return apply_public_heuristic_bias(
            scores,
            raw_scores,
            scale=scale,
            family_ids=family_ids,
            bias_family_ids=self._public_heuristic_bias_family_ids,
        )

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

    def _score_packed_public_heuristic_chunked(
        self,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> Tensor:
        if scoring_plan.candidate_count == 0:
            return torch.zeros((0,), device=scoring_plan.row_indices.device, dtype=dtype)
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            scores_chunks.append(
                self._score_packed_public_heuristic_plan(
                    scoring_plan.slice(start, end),
                    observation_context,
                    dtype=dtype,
                    scoring_profile=scoring_profile,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _score_packed_public_heuristic_plan(
        self,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> Tensor:
        row_indices_long = scoring_plan.row_indices.to(dtype=torch.long)
        candidate_count = scoring_plan.candidate_count
        score0 = torch.full((candidate_count,), -1000.0, dtype=dtype, device=row_indices_long.device)
        score1 = torch.zeros((candidate_count,), dtype=dtype, device=row_indices_long.device)
        score2 = torch.zeros((candidate_count,), dtype=dtype, device=row_indices_long.device)

        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]
        self_level_count = observation_context["self_level_count"].to(device=row_indices_long.device, dtype=dtype)
        self_clock_count = observation_context["self_clock_count"].to(device=row_indices_long.device, dtype=dtype)
        choice_page_start = observation_context["choice_page_start"].to(device=row_indices_long.device, dtype=dtype)
        choice_total = observation_context["choice_total"].to(device=row_indices_long.device, dtype=dtype)

        attackers_available, front_defenders = public_attack_profile(
            self_stage_numeric,
            opponent_stage_numeric,
            front_row_count=len(PUBLIC_HEURISTIC_FRONT_ROW_SLOTS),
            dtype=dtype,
        )

        (
            play_indices,
            hand_indices,
            move_indices,
            attack_indices,
            slot_family_indices,
            index_family_indices,
            default_indices,
        ) = self._partition_candidate_family_indices(scoring_plan.family_ids)

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
            attack_source_context, attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            del attack_source_context
            _defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                opponent_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            del _defender_context
            slot_pref = self._slot_preference_values(attack_slot_values, dtype=dtype)
            attacker_power = torch.clamp(attack_source_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
            attacker_soul = torch.clamp(attack_source_numeric[:, 5].to(dtype=dtype) * 4.0, min=0.0)
            defender_power = torch.clamp(defender_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
            attacker_occupied = attack_source_numeric[:, 0].to(dtype=dtype) > 0.5
            defender_occupied = defender_numeric[:, 0].to(dtype=dtype) > 0.5
            side_attack_allowed = attack_source_numeric[:, 6].to(dtype=dtype) > 0.5
            type_score = torch.zeros(attack_slot_values.shape, dtype=dtype, device=row_indices_long.device)
            if self._direct_attack_type_id >= 0:
                direct = attack_type_values == self._direct_attack_type_id
                type_score = torch.where(
                    direct,
                    torch.where(
                        defender_occupied,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_direct_blocked_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_direct_open_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            if self._frontal_attack_type_id >= 0:
                frontal = attack_type_values == self._frontal_attack_type_id
                type_score = torch.where(
                    frontal,
                    torch.where(
                        attacker_power >= defender_power,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_frontal_win_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_frontal_loss_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            if self._side_attack_type_id >= 0:
                side = attack_type_values == self._side_attack_type_id
                type_score = torch.where(
                    side,
                    torch.where(
                        side_attack_allowed,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_side_allowed_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_side_blocked_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            attack_score = (
                type_score
                + slot_pref
                + (attacker_soul * float(scoring_profile.attack_soul_scale))
                + torch.floor(attacker_power / 1000.0)
            )
            attack_score = torch.where(
                attacker_occupied,
                attack_score,
                attack_slot_values.new_full(attack_slot_values.shape, -1000.0, dtype=dtype),
            )
            score0.index_fill_(0, attack_indices, float(scoring_profile.attack_priority))
            score1.index_copy_(0, attack_indices, attack_score)

        if slot_family_indices.numel() > 0:
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

        if play_indices.numel() > 0:
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
                    play_stage_slots.new_full(
                        play_stage_slots.shape, float(scoring_profile.play_back_bonus), dtype=dtype
                    ),
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

        if hand_indices.numel() > 0:
            hand_rows = row_indices_long.index_select(0, hand_indices)
            hand_family_ids = scoring_plan.family_ids.index_select(0, hand_indices)
            hand_indices_values = scoring_plan.arg0.index_select(0, hand_indices)
            if self._climax_play_family_id >= 0:
                climax_mask = hand_family_ids == self._climax_play_family_id
                if bool(climax_mask.any().item()):
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
                    score2.index_copy_(
                        0, climax_indices, self._public_prefer_lower(hand_indices_values[climax_mask], dtype=dtype)
                    )
            if self._clock_from_hand_family_id >= 0:
                clock_mask = hand_family_ids == self._clock_from_hand_family_id
                if bool(clock_mask.any().item()):
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
                            clock_counts.new_full(
                                clock_counts.shape, float(scoring_profile.late_clock_score), dtype=dtype
                            ),
                        ),
                    )
                    score2.index_copy_(
                        0, clock_indices, self._public_prefer_lower(hand_indices_values[clock_mask], dtype=dtype)
                    )
            if self._main_event_family_id >= 0:
                event_mask = hand_family_ids == self._main_event_family_id
                if bool(event_mask.any().item()):
                    event_indices = hand_indices[event_mask]
                    score0.index_fill_(0, event_indices, float(scoring_profile.event_priority))
                    score1.index_fill_(0, event_indices, 10.0)
                    score2.index_copy_(
                        0, event_indices, self._public_prefer_lower(hand_indices_values[event_mask], dtype=dtype)
                    )
            if self._mulligan_select_family_id >= 0:
                mulligan_mask = hand_family_ids == self._mulligan_select_family_id
                if bool(mulligan_mask.any().item()):
                    mulligan_indices = hand_indices[mulligan_mask]
                    score0.index_fill_(0, mulligan_indices, float(scoring_profile.mulligan_select_priority))
                    score1.index_copy_(
                        0, mulligan_indices, self._public_prefer_lower(hand_indices_values[mulligan_mask], dtype=dtype)
                    )

        if index_family_indices.numel() > 0:
            index_rows = row_indices_long.index_select(0, index_family_indices)
            index_family_ids = scoring_plan.family_ids.index_select(0, index_family_indices)
            index_values = scoring_plan.arg0.index_select(0, index_family_indices)
            if self._choice_select_family_id >= 0:
                choice_mask = index_family_ids == self._choice_select_family_id
                if bool(choice_mask.any().item()):
                    choice_indices = index_family_indices[choice_mask]
                    score0.index_fill_(0, choice_indices, float(scoring_profile.choice_select_priority))
                    score1.index_copy_(
                        0, choice_indices, self._public_prefer_lower(index_values[choice_mask], dtype=dtype)
                    )
            if self._level_up_family_id >= 0:
                level_up_mask = index_family_ids == self._level_up_family_id
                if bool(level_up_mask.any().item()):
                    level_indices = index_family_indices[level_up_mask]
                    score0.index_fill_(0, level_indices, float(scoring_profile.level_up_priority))
                    score1.index_copy_(
                        0, level_indices, self._public_prefer_lower(index_values[level_up_mask], dtype=dtype)
                    )
            if self._trigger_order_family_id >= 0:
                trigger_mask = index_family_ids == self._trigger_order_family_id
                if bool(trigger_mask.any().item()):
                    trigger_indices = index_family_indices[trigger_mask]
                    score0.index_fill_(0, trigger_indices, float(scoring_profile.trigger_order_priority))
                    score1.index_copy_(
                        0, trigger_indices, self._public_prefer_lower(index_values[trigger_mask], dtype=dtype)
                    )
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
                            choice_total.index_select(0, next_rows)
                            - (choice_page_start.index_select(0, next_rows) + 16.0),
                            min=0.0,
                        ),
                    )
            if self._prev_page_family_id >= 0:
                prev_mask = index_family_ids == self._prev_page_family_id
                if bool(prev_mask.any().item()):
                    prev_indices = index_family_indices[prev_mask]
                    prev_rows = index_rows[prev_mask]
                    score0.index_fill_(0, prev_indices, float(scoring_profile.pager_priority))
                    score1.index_copy_(
                        0, prev_indices, torch.clamp(choice_page_start.index_select(0, prev_rows), min=0.0)
                    )

        if move_indices.numel() > 0:
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
                (
                    (move_to_slots == PUBLIC_HEURISTIC_CENTER_SLOT) & (move_from_slots != PUBLIC_HEURISTIC_CENTER_SLOT)
                ).to(dtype=dtype)
                * float(scoring_profile.move_center_bonus)
            )
            legal = (move_source_numeric[:, 0].to(dtype=dtype) > 0.5) & (
                move_target_numeric[:, 0].to(dtype=dtype) <= 0.5
            )
            move_score = torch.where(
                legal,
                (target_pref - source_pref) + bonus,
                move_to_slots.new_full(move_to_slots.shape, -1000.0, dtype=dtype),
            )
            score0.index_fill_(0, move_indices, float(scoring_profile.move_priority))
            score1.index_copy_(0, move_indices, move_score)

        if default_indices.numel() > 0:
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

        return self._combine_public_heuristic_scores(score0, score1, score2, dtype=dtype)
