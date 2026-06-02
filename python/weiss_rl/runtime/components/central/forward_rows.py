"""Dense central actor-row forwarding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.central.legal_hooks import concatenate_batch_legal_actions
from weiss_rl.runtime.components.central.model_hooks import actor_inference_model, model_accepts_legal_actions_kwarg

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralForwardRowsMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device
        action_dim: int

    def _central_forward_all_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch] | None,
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if not actors:
            return
        obs_concat = np.concatenate(obs_steps, axis=0)
        actor_concat = np.concatenate(actor_steps, axis=0)
        hidden_concat = torch.cat([actor.seat_hidden for actor in actors], dim=0)
        model = actor_inference_model(actors[0])
        legal_actions = None
        if (
            bool(getattr(model, "supports_legal_candidate_scoring", False))
            and batches is not None
            and model_accepts_legal_actions_kwarg(model)
        ):
            legal_actions = concatenate_batch_legal_actions(batches, action_space=int(self.action_dim))
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            obs_tensor = torch.as_tensor(obs_concat, device=self._device)
            actor_tensor = torch.as_tensor(actor_concat, device=self._device, dtype=torch.long)
            if legal_actions is None:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                )
            else:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                    legal_actions=legal_actions,
                )
        logits_concat = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, logits_out, values_out in zip(actors, logits_outs, values_outs, strict=True):
            count = int(logits_out.shape[0])
            logits_out[...] = logits_concat[offset : offset + count]
            values_out[...] = values_concat[offset : offset + count]
            actor.seat_hidden[...] = next_hidden_tensor[offset : offset + count]
            offset += count


__all__ = ["QueueRuntimeCentralForwardRowsMixin"]
