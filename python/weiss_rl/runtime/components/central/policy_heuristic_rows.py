"""Heuristic-backed central packed policy-row sampling."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.central.legal_hooks import optional_legal_action_meta, require_ids_offsets
from weiss_rl.runtime.components.policy_inference.heuristic_actor_outputs import write_heuristic_actor_outputs_ids

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralPolicyHeuristicRowsMixin:
    def _central_sample_policy_rows_ids_heuristic(
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
        heuristic_policy = self._teacher_policy
        if heuristic_policy is None:
            raise RuntimeError("heuristic actor policy backend requires an initialized teacher policy")
        model_started = time.perf_counter()
        if bool(getattr(self, "_actor_behavior_values_required", True)):
            self._central_value_and_advance_actor_rows(
                actors=actors,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
            )
        else:
            if self._should_track_heuristic_actor_hidden_state():
                self._central_advance_actor_rows(
                    actors=actors,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    row_indices_by_actor=row_indices_by_actor,
                )
            for actor_index, row_indices in enumerate(row_indices_by_actor):
                if row_indices.size:
                    values_outs[actor_index][row_indices] = 0.0
        self._record_batch_timer_ms("central_focal_policy_model", time.perf_counter() - model_started)
        scatter_started = time.perf_counter()
        for actor_index, (actor, batch, obs_step, row_indices) in enumerate(
            zip(actors, batches, obs_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            legal_ids, legal_offsets = require_ids_offsets(batch)
            legal_action_meta = self._ensure_legal_action_meta(legal_ids, optional_legal_action_meta(batch))
            chosen_actions = self._heuristic_public_actions_from_ids(
                actor=actor,
                heuristic_policy=heuristic_policy,
                row_indices=row_indices,
                obs_step=obs_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
            )
            self._maybe_debug_validate_sampled_packed_actions(
                source_label="central:focal:heuristic",
                row_indices=row_indices,
                action_subset=np.asarray(chosen_actions, dtype=np.int64),
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
            )
            write_heuristic_actor_outputs_ids(
                logits_out=None,
                row_indices=row_indices,
                chosen_actions=chosen_actions,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                actions_out=actions_outs[actor_index],
                logp_out=logp_outs[actor_index],
            )
        self._record_batch_timer_ms("central_focal_policy_scatter", time.perf_counter() - scatter_started)


__all__ = ["QueueRuntimeCentralPolicyHeuristicRowsMixin"]
