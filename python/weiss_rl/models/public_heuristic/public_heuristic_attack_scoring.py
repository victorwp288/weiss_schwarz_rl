"""Attack-candidate scoring for the public heuristic policy head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.public_heuristic.profiles import HeuristicPublicScoringProfile


class StructuredPublicHeuristicAttackScoringMixin:
    """Scores attack candidates from public board state and attack type."""

    def _score_public_attack_candidates(
        self: Any,
        *,
        scoring_plan: PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        self_stage_numeric: Tensor,
        opponent_stage_numeric: Tensor,
        row_indices_long: Tensor,
        attack_indices: Tensor,
        score0: Tensor,
        score1: Tensor,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> None:
        if attack_indices.numel() == 0:
            return
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
