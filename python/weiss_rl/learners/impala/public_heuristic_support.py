"""IMPALA public-heuristic target scoring support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import (
    active_public_heuristic_profiles,
    score_public_heuristic_target_logits,
)


class ImpalaPublicHeuristicSupportMixin:
    def _score_public_heuristic_target_logits(
        self: Any,
        *,
        forward_model: Any,
        obs_rows: Tensor,
        legal_actions: Any,
        observation_context: Mapping[str, Tensor] | None,
        device: torch.device,
    ) -> Tensor:
        return score_public_heuristic_target_logits(
            forward_model=forward_model,
            obs_rows=obs_rows,
            legal_actions=legal_actions,
            observation_context=observation_context,
            profiles=self.teacher_public_heuristic_profiles,
            profile_mode=self.teacher_public_heuristic_profile_mode,
            update_count=int(self.update_count),
            end_updates=int(self.teacher_public_heuristic_profiles_end_updates),
            temperature=float(self.teacher_public_heuristic_temperature),
            device=device,
        )

    def _active_teacher_public_heuristic_profiles(self: Any) -> tuple[str, ...]:
        return active_public_heuristic_profiles(
            self.teacher_public_heuristic_profiles,
            update_count=int(self.update_count),
            end_updates=int(self.teacher_public_heuristic_profiles_end_updates),
        )

    def _packed_public_heuristic_target_logits(
        self: Any,
        *,
        forward_model: Any,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        observation_context: Mapping[str, Tensor] | None,
    ) -> Tensor | None:
        total_rows = int(obs.shape[0] * obs.shape[1])
        active_rows = torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            return None
        flat_obs = obs.reshape(total_rows, obs.shape[-1])
        if int(active_rows.shape[0]) == total_rows:
            legal_actions = self._packed_legal_action_view(packed_legal)
            return self._score_public_heuristic_target_logits(
                forward_model=forward_model,
                obs_rows=flat_obs,
                legal_actions=legal_actions,
                observation_context=observation_context,
                device=flat_obs.device,
            )
        subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
        subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
        subset_obs = flat_obs.index_select(0, active_rows)
        subset_context = (
            None
            if observation_context is None
            else self._subset_observation_context_rows(
                observation_context,
                active_rows,
                row_count=total_rows,
            )
        )
        subset_target_logits = self._score_public_heuristic_target_logits(
            forward_model=forward_model,
            obs_rows=subset_obs,
            legal_actions=subset_legal_actions,
            observation_context=subset_context,
            device=subset_obs.device,
        )
        return self._scatter_packed_candidate_values(
            packed_legal,
            active_rows,
            subset_target_logits,
            fill_value=0.0,
        )


__all__ = ["ImpalaPublicHeuristicSupportMixin"]
