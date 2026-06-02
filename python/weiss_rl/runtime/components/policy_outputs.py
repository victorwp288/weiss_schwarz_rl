"""Policy-output filling adapters for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.runtime.components.opponent_context import opponent_context_indices_for_model

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimePolicyOutputMixin:
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _fill_policy_outputs_mask(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
        source_label: str = "policy_rows",
    ) -> None:
        del source_label
        focal_indices = np.flatnonzero(focal_rows)
        model_focal_indices, heuristic_focal_indices = self._split_focal_actor_rows(
            actor=actor,
            focal_indices=focal_indices,
            rng=rng,
        )
        if model_focal_indices.size:
            self._apply_policy_rows_mask(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=model_focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
                source_label="focal:model",
            )
        if heuristic_focal_indices.size:
            self._apply_heuristic_actor_rows_mask(
                actor=actor,
                row_indices=heuristic_focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_mask(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_mask=legal_mask,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )

    def _fill_policy_outputs_ids(
        self,
        *,
        actor: _ActorState,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        focal_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
    ) -> None:
        focal_indices = np.flatnonzero(focal_rows)
        model_focal_indices, heuristic_focal_indices = self._split_focal_actor_rows(
            actor=actor,
            focal_indices=focal_indices,
            rng=rng,
        )
        if model_focal_indices.size:
            focal_model = _actor_inference_model(actor)
            opponent_context_index = opponent_context_indices_for_model(
                focal_model,
                actor.opponent_policy_id_by_env.tolist(),
                batch_size=actor.opponent_policy_id_by_env.shape[0],
            )
            self._apply_policy_rows_ids(
                model=focal_model,
                hidden_state=actor.seat_hidden,
                row_indices=model_focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                opponent_context_index=opponent_context_index,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )
        if heuristic_focal_indices.size:
            self._apply_heuristic_actor_rows_ids(
                actor=actor,
                row_indices=heuristic_focal_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
            )
        opponent_indices = np.flatnonzero(~focal_rows)
        if opponent_indices.size:
            self._apply_opponent_rows_ids(
                actor=actor,
                row_indices=opponent_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
                logits_out=logits_out,
                values_out=values_out,
                actions_out=actions_out,
                logp_out=logp_out,
                rng=rng,
                sample_actions=sample_actions,
            )
