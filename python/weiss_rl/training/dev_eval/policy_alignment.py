"""Policy-alignment diagnostics for periodic development evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.heuristic_public.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.alignment import PolicyAlignmentAccumulator
from weiss_rl.eval.sampling.model_action_surface import ModelActionSurfaceSettings, model_action_surface_batch_and_ids
from weiss_rl.eval.sampling.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.eval.simulator.harness import ScheduledGame
from weiss_rl.model import PolicyValueModel


class PeriodicDevEvalPolicyAlignment:
    def __init__(
        self,
        *,
        model: PolicyValueModel,
        heuristic_policy: HeuristicPublicPolicy | None,
        focal_policy_id: str,
        opponent_policy_id: str,
        action_dim: int,
        device: torch.device,
        model_action_surface: ModelActionSurfaceSettings,
    ) -> None:
        self.model = model
        self.heuristic_policy = heuristic_policy
        self.focal_policy_id = focal_policy_id
        self.opponent_policy_id = opponent_policy_id
        self.action_dim = int(action_dim)
        self.device = device
        self.model_action_surface = model_action_surface
        action_catalog = getattr(model, "action_catalog", None)
        self._all = None if heuristic_policy is None else PolicyAlignmentAccumulator(action_catalog=action_catalog)
        self._focal_turns = (
            None if heuristic_policy is None else PolicyAlignmentAccumulator(action_catalog=action_catalog)
        )
        self._opponent_turns = (
            None if heuristic_policy is None else PolicyAlignmentAccumulator(action_catalog=action_catalog)
        )

    def initial_hidden(self) -> torch.Tensor | None:
        if self.heuristic_policy is None:
            return None
        return self.model.initial_seat_hidden(1, device=self.device)

    def record(
        self,
        *,
        batch: DecisionBoundaryBatch,
        scheduled_game: ScheduledGame,
        current_policy_id: str,
        current_seat: int,
        legal_ids: np.ndarray,
        alignment_hidden: torch.Tensor | None,
        action_sequence_state: Any | None = None,
    ) -> torch.Tensor | None:
        if (
            self.heuristic_policy is None
            or alignment_hidden is None
            or self._all is None
            or self._focal_turns is None
            or self._opponent_turns is None
        ):
            return alignment_hidden
        with torch.inference_mode():
            filtered_batch, filtered_legal_ids = model_action_surface_batch_and_ids(
                model=self.model,
                batch=batch,
                legal_ids=legal_ids,
                settings=self.model_action_surface,
                action_sequence_state=action_sequence_state,
            )
            model_logits, next_alignment_hidden = model_eval_logits_for_legal_ids(
                model=self.model,
                batch=filtered_batch,
                current_seat=int(current_seat),
                seat_hidden=alignment_hidden,
                legal_ids=filtered_legal_ids,
                action_dim=self.action_dim,
                device=self.device,
            )
        reference_action = int(
            self.heuristic_policy.choose_action(
                np.asarray(filtered_batch.obs[0], dtype=np.float32),
                filtered_legal_ids,
            )
        )
        self._all.add(
            model_logits=model_logits,
            legal_ids=filtered_legal_ids,
            reference_action_id=reference_action,
        )
        if current_policy_id == self.focal_policy_id:
            self._focal_turns.add(
                model_logits=model_logits,
                legal_ids=filtered_legal_ids,
                reference_action_id=reference_action,
            )
        elif current_policy_id == scheduled_game.opponent_policy_id:
            self._opponent_turns.add(
                model_logits=model_logits,
                legal_ids=filtered_legal_ids,
                reference_action_id=reference_action,
            )
        return next_alignment_hidden

    def summary(self) -> dict[str, Any] | None:
        if (
            self.heuristic_policy is None
            or self._all is None
            or self._focal_turns is None
            or self._opponent_turns is None
        ):
            return None
        return {
            "schema": "policy_alignment_diagnostics_v1",
            "model_policy_id": self.focal_policy_id,
            "reference_policy_id": self.opponent_policy_id,
            "reference_kind": "heuristic_public",
            "legal_surface": "model_action_surface_filtered_legal_ids",
            "all_decisions": self._all.summary(),
            "focal_policy_turns": self._focal_turns.summary(),
            "opponent_policy_turns": self._opponent_turns.summary(),
        }


__all__ = ["PeriodicDevEvalPolicyAlignment"]
