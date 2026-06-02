"""Central value and hidden-state row forwarding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from weiss_rl.runtime.components.central.model_hooks import actor_inference_model

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralValueRowsMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device

    def _central_value_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = actor_inference_model(actors[0])
        for actor_index, (actor, obs_step, actor_step, row_indices) in enumerate(
            zip(actors, obs_steps, actor_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor_index, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            value_seat_aware = getattr(model, "value_seat_aware", None)
            if callable(value_seat_aware):
                value_tensor = value_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, value_tensor, _next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        offset = 0
        for actor_index, row_indices in entries:
            count = int(row_indices.shape[0])
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            offset += count

    def _central_value_and_advance_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, _ActorState, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = actor_inference_model(actors[0])
        for actor_index, (actor, obs_step, actor_step, row_indices) in enumerate(
            zip(actors, obs_steps, actor_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor_index, actor, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        obs_tensor = torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device)
        actor_tensor = torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            forward_trunk = getattr(model, "forward_trunk_packed_seat_aware", None)
            if callable(forward_trunk):
                _recurrent_output, _state_repr, _observation_context, value_tensor, next_hidden = forward_trunk(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                )
            else:
                value_seat_aware = getattr(model, "value_seat_aware", None)
                advance_hidden = getattr(model, "advance_seat_hidden", None)
                if callable(value_seat_aware) and callable(advance_hidden):
                    value_tensor = value_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
                    next_hidden = advance_hidden(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
                else:
                    _logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor_index, actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            offset += count

    def _central_advance_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[_ActorState, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = actor_inference_model(actors[0])
        for actor, obs_step, actor_step, row_indices in zip(
            actors,
            obs_steps,
            actor_steps,
            row_indices_by_actor,
            strict=True,
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            advance_hidden = getattr(model, "advance_seat_hidden", None)
            if callable(advance_hidden):
                next_hidden = advance_hidden(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            offset += count


__all__ = ["QueueRuntimeCentralValueRowsMixin"]
