"""Trunk forwarding helpers for structured policy/value models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor


class StructuredPolicyValueTrunkMixin:
    if TYPE_CHECKING:
        _compiled_trunk_packed_core: Any | None
        _compiled_trunk_sequence_core: Any | None
        _trunk_compile_last_error: str | None

    def enable_trunk_compile(self: Any, *, mode: str = "reduce-overhead") -> Any:
        compiled_packed = self._compiled_trunk_packed_core
        compiled_sequence = self._compiled_trunk_sequence_core
        if compiled_packed is None:
            compiled_packed = torch.compile(
                self._forward_trunk_packed_core,
                mode=mode,
            )
        if compiled_sequence is None:
            compiled_sequence = torch.compile(
                self._forward_trunk_sequence_core,
                mode=mode,
            )
        self._compiled_trunk_packed_core = compiled_packed
        self._compiled_trunk_sequence_core = compiled_sequence
        return self

    def forward_trunk_packed_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        trunk_forward = self._compiled_trunk_packed_core
        has_context = opponent_context_index is not None
        if trunk_forward is not None and not has_context:
            try:
                recurrent_output, obs_batch, value, next_seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_packed_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    opponent_context_index=opponent_context_index,
                )
        else:
            recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                obs,
                acting_seat,
                seat_hidden_state,
                opponent_context_index=opponent_context_index,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(recurrent_output, obs=obs_batch)
        return recurrent_output, state_repr, observation_context, value, next_seat_hidden

    def forward_trunk_sequence_seat_aware(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        time_steps = int(obs.shape[0])
        batch_size = int(obs.shape[1])
        trunk_forward = self._compiled_trunk_sequence_core
        has_resets = reset_before_step is not None and bool(torch.as_tensor(reset_before_step).any().item())
        has_context = opponent_context_index is not None
        if trunk_forward is not None and not has_resets and not has_context:
            try:
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_sequence_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    reset_before_step=reset_before_step,
                    opponent_context_index=opponent_context_index,
                )
        else:
            recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                obs,
                acting_seat,
                seat_hidden_state,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(
            recurrent_flat,
            obs=flat_obs_batch,
        )
        return recurrent_flat, state_repr, observation_context, value_flat.reshape(time_steps, batch_size), seat_hidden

    def _forward_trunk_packed_core(
        self: Any,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return recurrent_output, obs_batch, value, next_seat_hidden

    def _forward_trunk_sequence_core(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        recurrent_flat, flat_obs_batch, seat_hidden, _time_steps, _batch_size = self._sequence_recurrent_outputs(
            obs,
            acting_seat,
            seat_hidden_state,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
        )
        value_flat = self.value_head(recurrent_flat).squeeze(-1)
        return recurrent_flat, flat_obs_batch, value_flat, seat_hidden

    def _sequence_recurrent_outputs(
        self: Any,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        time_steps, batch_size, obs_dim = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        encoded_flat = self.encode(flat_obs)
        encoded = encoded_flat.reshape(time_steps, batch_size, encoded_flat.shape[-1])
        seat_hidden = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded[0],
        )
        reset_mask = None
        if reset_before_step is not None:
            reset_mask = torch.as_tensor(reset_before_step, device=encoded.device, dtype=torch.bool)
            if reset_mask.ndim != 2 or reset_mask.shape != obs.shape[:2]:
                raise ValueError("reset_before_step must be 2D (time, batch) with the same leading dimensions as obs")
        context_index = None
        if opponent_context_index is not None:
            context_index = torch.as_tensor(opponent_context_index, device=encoded.device, dtype=torch.long)
            if context_index.ndim != 2 or context_index.shape != obs.shape[:2]:
                raise ValueError(
                    "opponent_context_index must be 2D (time, batch) with the same leading dimensions as obs"
                )
        recurrent_steps: list[Tensor] = []
        for step_index, (step_encoded, step_seat) in enumerate(
            zip(encoded.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)
        ):
            if reset_mask is not None:
                step_reset = reset_mask[step_index]
                if bool(step_reset.any().item()):
                    reset_rows = torch.nonzero(step_reset, as_tuple=False).squeeze(1)
                    seat_hidden = seat_hidden.clone()
                    seat_hidden.index_copy_(
                        0,
                        reset_rows,
                        self.initial_seat_hidden(
                            int(reset_rows.numel()),
                            device=seat_hidden.device,
                            dtype=seat_hidden.dtype,
                            opponent_context_indices=(
                                None if context_index is None else context_index[step_index].index_select(0, reset_rows)
                            ),
                        ),
                    )
            recurrent_output, seat_hidden = self.recurrent_step_seat_aware(
                step_encoded,
                step_seat,
                seat_hidden,
            )
            recurrent_output = self._apply_opponent_context_recurrent_adapter(
                recurrent_output,
                None if context_index is None else context_index[step_index],
            )
            recurrent_steps.append(recurrent_output)
        recurrent = torch.stack(recurrent_steps, dim=0)
        recurrent_flat = recurrent.reshape(time_steps * batch_size, recurrent.shape[-1])
        return recurrent_flat, self._require_observation_batch(flat_obs), seat_hidden, time_steps, batch_size


__all__ = ["StructuredPolicyValueTrunkMixin"]
