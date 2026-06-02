"""Central packed policy-row routing helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.central.forward_rows import QueueRuntimeCentralForwardRowsMixin
from weiss_rl.runtime.components.central.policy_heuristic_rows import QueueRuntimeCentralPolicyHeuristicRowsMixin
from weiss_rl.runtime.components.central.policy_model_rows import QueueRuntimeCentralPolicyModelRowsMixin
from weiss_rl.runtime.components.central.value_rows import QueueRuntimeCentralValueRowsMixin

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralRowsMixin(
    QueueRuntimeCentralValueRowsMixin,
    QueueRuntimeCentralForwardRowsMixin,
    QueueRuntimeCentralPolicyModelRowsMixin,
    QueueRuntimeCentralPolicyHeuristicRowsMixin,
):
    if TYPE_CHECKING:
        action_dim: int

    def _central_sample_policy_rows_ids(
        self: Any,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
        actions_outs: Sequence[np.ndarray],
        logp_outs: Sequence[np.ndarray],
    ) -> None:
        if self._actor_policy_backend != "heuristic_public":
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        heuristic_fraction = self._active_actor_heuristic_fraction()
        if heuristic_fraction >= 1.0:
            self._central_sample_policy_rows_ids_heuristic(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        if heuristic_fraction <= 0.0:
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        heuristic_rows_by_actor: list[np.ndarray] = []
        model_rows_by_actor: list[np.ndarray] = []
        any_heuristic_rows = False
        any_model_rows = False
        for actor, row_indices in zip(actors, row_indices_by_actor, strict=True):
            if row_indices.size == 0:
                heuristic_rows_by_actor.append(row_indices)
                model_rows_by_actor.append(row_indices)
                continue
            heuristic_mask = actor.rng.random(row_indices.shape[0]) < heuristic_fraction
            heuristic_rows = row_indices[heuristic_mask]
            model_rows = row_indices[~heuristic_mask]
            heuristic_rows_by_actor.append(heuristic_rows)
            model_rows_by_actor.append(model_rows)
            any_heuristic_rows = any_heuristic_rows or heuristic_rows.size > 0
            any_model_rows = any_model_rows or model_rows.size > 0
        if any_heuristic_rows:
            self._central_sample_policy_rows_ids_heuristic(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=heuristic_rows_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
        if any_model_rows:
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=model_rows_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
