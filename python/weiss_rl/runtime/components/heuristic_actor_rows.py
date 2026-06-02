"""QueueRuntime heuristic actor row application helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.runtime.components.policy_inference.heuristic_actor_outputs import (
    write_heuristic_actor_outputs_ids,
    write_heuristic_actor_outputs_mask,
)

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeHeuristicActorRowsMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device
        action_dim: int

    def _advance_hidden_only(
        self: Any,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
    ) -> None:
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
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                )
            else:
                _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                )
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()

    def _should_track_heuristic_actor_hidden_state(self: Any) -> bool:
        return bool(getattr(self, "_heuristic_actor_hidden_state_tracking", True))

    def _value_and_advance_rows(
        self: Any,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        values_out: np.ndarray,
    ) -> None:
        if row_indices.size == 0:
            return
        obs_tensor = torch.as_tensor(obs_step[row_indices], device=self._device)
        actor_tensor = torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long)
        hidden_tensor = hidden_state[row_indices]
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
                    hidden_tensor,
                )
            else:
                value_seat_aware = getattr(model, "value_seat_aware", None)
                advance_hidden = getattr(model, "advance_seat_hidden", None)
                if callable(value_seat_aware) and callable(advance_hidden):
                    value_tensor = value_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_tensor,
                    )
                    next_hidden = advance_hidden(
                        obs_tensor,
                        actor_tensor,
                        hidden_tensor,
                    )
                else:
                    _logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_tensor,
                    )
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()
        values_out[row_indices] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def _apply_heuristic_actor_rows_mask(
        self: Any,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
    ) -> None:
        heuristic_policy = self._teacher_policy
        if heuristic_policy is None:
            raise RuntimeError("heuristic actor policy backend requires an initialized teacher policy")
        if bool(getattr(self, "_actor_behavior_values_required", True)):
            self._value_and_advance_rows(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=row_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                values_out=values_out,
            )
        else:
            if self._should_track_heuristic_actor_hidden_state():
                self._advance_hidden_only(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=row_indices,
                    obs_step=obs_step,
                    actor_step=actor_step,
                )
            values_out[row_indices] = 0.0
        chosen_actions = self._heuristic_public_actions_from_mask(
            actor=actor,
            heuristic_policy=heuristic_policy,
            row_indices=row_indices,
            obs_step=obs_step,
            legal_mask=legal_mask,
        )
        write_heuristic_actor_outputs_mask(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_mask=legal_mask,
            actions_out=actions_out,
            logp_out=logp_out,
            action_dim=int(self.action_dim),
        )

    def _apply_heuristic_actor_rows_ids(
        self: Any,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
    ) -> None:
        heuristic_policy = self._teacher_policy
        if heuristic_policy is None:
            raise RuntimeError("heuristic actor policy backend requires an initialized teacher policy")
        if bool(getattr(self, "_actor_behavior_values_required", True)):
            self._value_and_advance_rows(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=row_indices,
                obs_step=obs_step,
                actor_step=actor_step,
                values_out=values_out,
            )
        else:
            if self._should_track_heuristic_actor_hidden_state():
                self._advance_hidden_only(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=row_indices,
                    obs_step=obs_step,
                    actor_step=actor_step,
                )
            values_out[row_indices] = 0.0
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
            source_label="focal:heuristic",
            row_indices=row_indices,
            action_subset=np.asarray(chosen_actions, dtype=np.int64),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
        write_heuristic_actor_outputs_ids(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            actions_out=actions_out,
            logp_out=logp_out,
        )


__all__ = ["QueueRuntimeHeuristicActorRowsMixin"]
