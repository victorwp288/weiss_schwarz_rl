"""Structured policy-head public heuristic scoring mixin."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.models.public_heuristic.public_heuristic_attack_scoring import StructuredPublicHeuristicAttackScoringMixin
from weiss_rl.models.public_heuristic.public_heuristic_candidate_groups import (
    StructuredPublicHeuristicCandidateGroupMixin,
)
from weiss_rl.models.public_heuristic.public_heuristic_primitives import StructuredPublicHeuristicPrimitiveMixin
from weiss_rl.models.public_heuristic.public_heuristic_slots import (
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
    public_attack_profile,
)
from weiss_rl.public_heuristic.profiles import HeuristicPublicScoringProfile


class StructuredPublicHeuristicScoringMixin(
    StructuredPublicHeuristicAttackScoringMixin,
    StructuredPublicHeuristicCandidateGroupMixin,
    StructuredPublicHeuristicPrimitiveMixin,
):
    """Score packed legal candidates with the public heuristic profile.

    The owning policy head provides the tensors, embeddings, and candidate
    partition helpers. Primitive head-state adapters live in
    `public_heuristic_primitives.py`; this class owns the packed scoring flow.
    """

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

        self._score_public_attack_candidates(
            scoring_plan=scoring_plan,
            observation_context=observation_context,
            self_stage_numeric=self_stage_numeric,
            opponent_stage_numeric=opponent_stage_numeric,
            row_indices_long=row_indices_long,
            attack_indices=attack_indices,
            score0=score0,
            score1=score1,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )

        self._score_public_slot_candidates(
            scoring_plan=scoring_plan,
            observation_context=observation_context,
            self_stage_numeric=self_stage_numeric,
            row_indices_long=row_indices_long,
            slot_family_indices=slot_family_indices,
            score0=score0,
            score1=score1,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_play_candidates(
            scoring_plan=scoring_plan,
            observation_context=observation_context,
            self_stage_numeric=self_stage_numeric,
            row_indices_long=row_indices_long,
            play_indices=play_indices,
            score0=score0,
            score1=score1,
            score2=score2,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_hand_candidates(
            scoring_plan=scoring_plan,
            hand_indices=hand_indices,
            row_indices_long=row_indices_long,
            attackers_available=attackers_available,
            front_defenders=front_defenders,
            self_level_count=self_level_count,
            self_clock_count=self_clock_count,
            score0=score0,
            score1=score1,
            score2=score2,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_index_candidates(
            scoring_plan=scoring_plan,
            index_family_indices=index_family_indices,
            row_indices_long=row_indices_long,
            choice_page_start=choice_page_start,
            choice_total=choice_total,
            score0=score0,
            score1=score1,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_move_candidates(
            scoring_plan=scoring_plan,
            observation_context=observation_context,
            self_stage_numeric=self_stage_numeric,
            row_indices_long=row_indices_long,
            move_indices=move_indices,
            score0=score0,
            score1=score1,
            dtype=dtype,
            scoring_profile=scoring_profile,
        )
        self._score_public_default_candidates(
            scoring_plan=scoring_plan,
            default_indices=default_indices,
            score0=score0,
            scoring_profile=scoring_profile,
        )

        return self._combine_public_heuristic_scores(score0, score1, score2, dtype=dtype)
