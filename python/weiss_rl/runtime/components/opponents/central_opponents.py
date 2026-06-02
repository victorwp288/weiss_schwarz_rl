"""Central opponent-output overwrite helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.opponents.central_heuristic_opponent_apply import (
    apply_central_heuristic_opponent_outputs,
)
from weiss_rl.runtime.components.opponents.central_opponent_groups import group_central_opponent_rows
from weiss_rl.runtime.components.opponents.central_snapshot_opponents import (
    apply_central_snapshot_opponent_policy,
)

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralOpponentMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _overwrite_central_outputs_with_configured_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "python_scalar":
            for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
                actors,
                batches,
                obs_steps,
                actor_steps,
                logits_outs,
                values_outs,
                strict=True,
            ):
                self._overwrite_central_outputs_with_batched_opponents(
                    actors=[actor],
                    batches=[batch],
                    obs_steps=[obs_step],
                    actor_steps=[actor_step],
                    logits_outs=[logits_out],
                    values_outs=[values_out],
                )
            return
        self._overwrite_central_outputs_with_batched_opponents(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            logits_outs=logits_outs,
            values_outs=values_outs,
        )

    def _overwrite_central_outputs_with_opponents(
        self,
        *,
        actor: _ActorState,
        batch: DecisionBoundaryBatch,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
    ) -> None:
        self._overwrite_central_outputs_with_configured_opponents(
            actors=[actor],
            batches=[batch],
            obs_steps=[obs_step],
            actor_steps=[actor_step],
            logits_outs=[logits_out],
            values_outs=[values_out],
        )

    def _overwrite_central_outputs_with_batched_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        import time

        overwrite_started = time.perf_counter()
        policy_groups = group_central_opponent_rows(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            logits_outs=logits_outs,
            values_outs=values_outs,
        )

        for policy_id, entries in sorted(policy_groups.items()):
            heuristic_policy = self._heuristic_opponent_policy(policy_id)
            if heuristic_policy is not None:
                apply_central_heuristic_opponent_outputs(
                    policy_id=policy_id,
                    entries=entries,
                    heuristic_policy=heuristic_policy,
                    fixed_opponent_backend=str(getattr(self, "_fixed_opponent_backend", "python_batched")),
                    track_heuristic_hidden_state=self._should_track_heuristic_actor_hidden_state(),
                    central_advance_actor_rows=self._central_advance_actor_rows,
                    heuristic_public_actions_from_ids=self._heuristic_public_actions_from_ids,
                    heuristic_public_actions_from_mask=self._heuristic_public_actions_from_mask,
                    ensure_legal_action_meta=self._ensure_legal_action_meta,
                    maybe_debug_validate_sampled_packed_actions=self._maybe_debug_validate_sampled_packed_actions,
                    write_deterministic_logits_from_packed=self._write_deterministic_logits_from_packed,
                    write_deterministic_logits=self._write_deterministic_logits,
                )
                continue
            config = getattr(self, "config", None)
            apply_central_snapshot_opponent_policy(
                policy_id=policy_id,
                entries=entries,
                opponent_models=self._opponent_models,
                opponent_model_locks=self._opponent_model_locks,
                device=self._device,
                amp_enabled=self._actor_amp_enabled,
                action_selection=str(getattr(config, "fixed_model_opponent_action_selection", "sample")),
                pass_action_id=int(getattr(config, "pass_action_id", 0)),
                action_dim=int(getattr(self, "action_dim", 0)),
                ensure_legal_action_meta=self._ensure_legal_action_meta,
            )
        self._record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
