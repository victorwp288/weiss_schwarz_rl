"""Action-selection helpers for simulator-backed evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.masking import assert_strictly_increasing_legal_ids
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.policies.resolution import ResolvedEvalPolicy
from weiss_rl.eval.sampling.model_action_surface import model_action_surface_batch_and_ids
from weiss_rl.eval.sampling.model_sampling import model_eval_logits_for_legal_ids
from weiss_rl.eval.sampling.rng_pcg32 import Pcg32XshRrV1
from weiss_rl.eval.simulator.harness import ScheduledGame, sample_action_pinned, select_action_argmax_pinned
from weiss_rl.eval.simulator.simulator_policy_step import EvalPolicyStep
from weiss_rl.runtime.components.opponent_context import (
    eval_policy_uses_opponent_context,
    initial_seat_hidden_for_opponents,
)


class SimulatorActionSelectionMixin:
    """Policy lookup, hidden-state, and legal-action handling for eval games."""

    if TYPE_CHECKING:
        policies: dict[str, ResolvedEvalPolicy]
        _baseline_logits: np.ndarray
        _device: torch.device
        _eval_sampling_algorithm: str
        _model_action_surface: Any
        _model_sampling_temperature: float
        action_dim: int
        pass_action_id: int
        require_sorted_legal_ids: bool

        def _should_run_god_search(
            self,
            *,
            policy: ResolvedEvalPolicy,
            current_policy_id: str,
            scheduled_game: ScheduledGame | None,
            legal_ids_for_model: np.ndarray,
            game_search_state: dict[str, int] | None,
        ) -> bool: ...

        def _select_action_with_god_search(
            self,
            *,
            scheduled_game: ScheduledGame | None,
            batch: DecisionBoundaryBatch,
            current_seat: int,
            current_policy_id: str,
            opponent_policy_id: str,
            root_seat_hidden: torch.Tensor,
            root_next_seat_hidden: torch.Tensor | None,
            seat_hidden_by_seat: Mapping[int, torch.Tensor | None] | None,
            action_sequence_state: Any | None,
            action_history: Sequence[int],
            root_logits: np.ndarray,
            legal_ids: np.ndarray,
            legal_ids_for_model: np.ndarray,
            base_action: int,
            game_search_state: dict[str, int] | None,
        ) -> int: ...

    def _select_action(
        self,
        *,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        opponent_policy_id: str,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
        scheduled_game: ScheduledGame | None = None,
        action_history: Sequence[int] = (),
        seat_hidden_by_seat: Mapping[int, torch.Tensor | None] | None = None,
        game_search_state: dict[str, int] | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        policy = self._require_eval_policy(current_policy_id)
        step = self._select_base_policy_step(
            policy=policy,
            current_policy_id=current_policy_id,
            opponent_policy_id=opponent_policy_id,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            rng=rng,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
        )
        action = step.action
        if step.has_model_surface and self._should_run_god_search(
            policy=policy,
            current_policy_id=current_policy_id,
            scheduled_game=scheduled_game,
            legal_ids_for_model=step.legal_ids_for_model,
            game_search_state=game_search_state,
        ):
            action = self._select_action_with_god_search(
                scheduled_game=scheduled_game,
                batch=batch,
                current_seat=current_seat,
                current_policy_id=current_policy_id,
                opponent_policy_id=opponent_policy_id,
                root_seat_hidden=seat_hidden,
                root_next_seat_hidden=step.next_seat_hidden,
                seat_hidden_by_seat=seat_hidden_by_seat,
                action_sequence_state=action_sequence_state,
                action_history=action_history,
                root_logits=step.logits,
                legal_ids=legal_ids,
                legal_ids_for_model=step.legal_ids_for_model,
                base_action=action,
                game_search_state=game_search_state,
            )
        return action, step.next_seat_hidden

    def _require_eval_policy(self, policy_id: str) -> ResolvedEvalPolicy:
        policy = self.policies.get(policy_id)
        if policy is None:
            raise RuntimeError(f"Missing resolved eval policy for {policy_id!r}")
        return policy

    def _select_base_policy_step(
        self,
        *,
        policy: ResolvedEvalPolicy,
        current_policy_id: str,
        opponent_policy_id: str,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
        sampling_algorithm: str | None = None,
    ) -> EvalPolicyStep:
        """Select the policy's own action before optional simulator search."""
        if policy.heuristic_policy is not None:
            action = policy.heuristic_policy.choose_action(
                np.asarray(batch.obs[0], dtype=np.float32),
                legal_ids,
            )
            return EvalPolicyStep(action=int(action), next_seat_hidden=seat_hidden)
        if policy.model is None:
            action, _logp = sample_action_pinned(
                self._baseline_logits,
                legal_ids,
                rng=rng,
            )
            return EvalPolicyStep(action=action, next_seat_hidden=seat_hidden)
        if seat_hidden is None:
            raise RuntimeError(f"Missing hidden state for eval policy {current_policy_id!r}")
        logits, next_seat_hidden, legal_ids_for_model = self._model_logits_for_eval(
            policy=policy,
            current_policy_id=current_policy_id,
            opponent_policy_id=opponent_policy_id,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
        )
        action, _logp = self._select_model_action_from_logits(
            logits=logits,
            legal_ids=legal_ids_for_model,
            rng=rng,
            sampling_algorithm=sampling_algorithm or self._eval_sampling_algorithm,
        )
        return EvalPolicyStep(
            action=action,
            next_seat_hidden=next_seat_hidden,
            logits=logits,
            legal_ids_for_model=legal_ids_for_model,
        )

    def _model_logits_for_eval(
        self,
        *,
        policy: ResolvedEvalPolicy,
        current_policy_id: str,
        opponent_policy_id: str,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        seat_hidden: torch.Tensor,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
    ) -> tuple[np.ndarray, torch.Tensor | None, np.ndarray]:
        batch_for_model, legal_ids_for_model = model_action_surface_batch_and_ids(
            model=policy.model,
            batch=batch,
            legal_ids=legal_ids,
            settings=self._model_action_surface,
            action_sequence_state=action_sequence_state,
        )
        if policy.model is None:
            raise RuntimeError(f"Missing model for eval policy {current_policy_id!r}")
        with torch.inference_mode():
            logits, next_seat_hidden = model_eval_logits_for_legal_ids(
                model=policy.model,
                batch=batch_for_model,
                current_seat=int(current_seat),
                seat_hidden=seat_hidden,
                legal_ids=legal_ids_for_model,
                action_dim=int(self.action_dim),
                device=self._device,
                opponent_context_index=self._opponent_context_index_for_eval(
                    policy=policy,
                    policy_id=current_policy_id,
                    opponent_policy_id=opponent_policy_id,
                ),
            )
        return logits, next_seat_hidden, legal_ids_for_model

    def _select_model_action_from_logits(
        self,
        *,
        logits: np.ndarray,
        legal_ids: np.ndarray,
        rng: Pcg32XshRrV1,
        sampling_algorithm: str,
    ) -> tuple[int, np.float32]:
        if sampling_algorithm == "model_argmax_pinned_v1":
            return select_action_argmax_pinned(
                logits,
                legal_ids,
                pass_action_id=self.pass_action_id,
            )
        if sampling_algorithm == "pinned_cdf_pcg_v1":
            return sample_action_pinned(
                logits,
                legal_ids,
                rng=rng,
                pass_action_id=self.pass_action_id,
                temperature=self._model_sampling_temperature,
            )
        raise ValueError(f"unsupported eval sampling algorithm: {sampling_algorithm!r}")

    def _select_action_without_god_search(
        self,
        *,
        batch: DecisionBoundaryBatch,
        current_seat: int,
        current_policy_id: str,
        opponent_policy_id: str,
        seat_hidden: torch.Tensor | None,
        rng: Pcg32XshRrV1,
        legal_ids: np.ndarray,
        action_sequence_state: Any | None = None,
        sampling_algorithm: str | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        step = self._select_base_policy_step(
            policy=self._require_eval_policy(current_policy_id),
            current_policy_id=current_policy_id,
            opponent_policy_id=opponent_policy_id,
            batch=batch,
            current_seat=current_seat,
            seat_hidden=seat_hidden,
            rng=rng,
            legal_ids=legal_ids,
            action_sequence_state=action_sequence_state,
            sampling_algorithm=sampling_algorithm or self._eval_sampling_algorithm,
        )
        return step.action, step.next_seat_hidden

    def _initial_hidden(self, policy_id: str, *, opponent_policy_id: str | None = None) -> torch.Tensor | None:
        policy = self.policies.get(policy_id)
        if policy is None or policy.model is None:
            return None
        if opponent_policy_id is not None and eval_policy_uses_opponent_context(policy.model, policy_id):
            return initial_seat_hidden_for_opponents(
                policy.model,
                1,
                device=self._device,
                opponent_policy_ids=[opponent_policy_id],
            )
        return initial_seat_hidden_for_opponents(policy.model, 1, device=self._device)

    def _opponent_context_index_for_eval(
        self,
        *,
        policy: ResolvedEvalPolicy,
        policy_id: str,
        opponent_policy_id: str,
    ) -> int | None:
        if policy.model is None or not eval_policy_uses_opponent_context(policy.model, policy_id):
            return None
        index_fn = getattr(policy.model, "opponent_context_indices_for_policy_ids", None)
        if not callable(index_fn):
            return None
        indices = index_fn([opponent_policy_id], batch_size=1)
        if not indices:
            return None
        return int(indices[0])

    def _legal_ids_for_env_row(self, *, batch: DecisionBoundaryBatch) -> np.ndarray:
        if batch.ids_offsets is None:
            raise RuntimeError("Pinned evaluation requires ids_offsets legality")
        legal_ids, legal_offsets = batch.ids_offsets
        row = np.asarray(legal_ids[int(legal_offsets[0]) : int(legal_offsets[1])], dtype=np.uint32)
        if self.require_sorted_legal_ids:
            assert_strictly_increasing_legal_ids(row)
        return row


__all__ = ["SimulatorActionSelectionMixin"]
