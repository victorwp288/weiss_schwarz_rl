"""Plan and chunk helpers for packed structured-policy scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.models.actions.action_plans import PackedScoringPlan
from weiss_rl.models.actions.candidate_partitioning import partition_candidate_family_indices
from weiss_rl.models.backbone.tensor_ops import optional_embedding, packed_row_indices
from weiss_rl.models.scoring.packed_legal_tensors import require_packed_legal_tensors
from weiss_rl.models.scoring.packed_projection import StructuredPackedProjectionMixin

_PackedScoringPlan = PackedScoringPlan
_optional_embedding = optional_embedding
_packed_row_indices = packed_row_indices


class StructuredPackedScoringSupportMixin(StructuredPackedProjectionMixin):
    """Packed scoring setup and chunking helpers used by the structured head."""

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
        packed = require_packed_legal_tensors(
            legal_actions,
            device=state_repr.device,
            row_count=int(state_repr.shape[0]),
            missing_message="projecting packed candidate representations requires packed ids and offsets",
        )
        if packed.candidate_count == 0:
            return state_repr.new_zeros((0, state_repr.shape[1]))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=packed.ids,
            offsets=packed.offsets,
            candidate_meta=packed.meta,
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


__all__ = ["StructuredPackedScoringSupportMixin"]
