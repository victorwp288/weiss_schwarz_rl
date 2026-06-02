"""IMPALA learner policy-anchor regularization support."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from weiss_rl.learners.policy_anchor import (
    clone_frozen_policy_anchor,
    packed_candidate_anchor_kl_loss,
    packed_candidate_anchor_top_action_loss,
)


def _batch_value(batch: Any, key: str) -> Any:
    # Resolve through impala_learner so the historical helper remains the compatibility hook.
    from weiss_rl.learners import impala_learner as learner_module

    return learner_module._batch_value(batch, key)


class ImpalaPolicyAnchorSupportMixin:
    model: nn.Module | None
    _policy_anchor_model: nn.Module | None

    def _ensure_policy_anchor_model(self: Any) -> nn.Module:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to create a policy anchor")
        if self._policy_anchor_model is None:
            self._policy_anchor_model = clone_frozen_policy_anchor(self.model)
        return self._policy_anchor_model

    def reset_policy_anchor_to_current_model(self: Any, *, force: bool = False) -> None:
        """Refresh the frozen anchor after externally replacing model weights."""

        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to reset a policy anchor")
        if not force and float(self.policy_anchor_coef) == 0.0 and float(self.policy_anchor_top_action_coef) == 0.0:
            self._policy_anchor_model = None
            return
        self._policy_anchor_model = clone_frozen_policy_anchor(self.model)

    def policy_anchor_state_dict(self: Any) -> dict[str, Tensor] | None:
        if self._policy_anchor_model is None:
            return None
        return self._policy_anchor_model.state_dict()

    def load_policy_anchor_state_dict(self: Any, state_dict: Mapping[str, Any] | None) -> None:
        if state_dict is None:
            self._policy_anchor_model = None
            return
        anchor_model = self._ensure_policy_anchor_model()
        anchor_model.load_state_dict(state_dict)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)

    def _factorized_candidate_log_probs_for_model(
        self: Any,
        forward_model: nn.Module,
        batch: Any,
        *,
        obs: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        reset_before_step: Tensor | None,
    ) -> Tensor:
        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            _batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            expected_shape=expected_shape,
        )
        if acting_seat is None:
            raise ValueError("policy-anchor regularization requires acting seat information")
        trunk_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
        opponent_context_index = _batch_value(batch, "opponent_context_index")
        if opponent_context_index is not None:
            opponent_context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
            if tuple(opponent_context_index.shape) != tuple(expected_shape):
                raise ValueError(
                    "opponent_context_index must match policy-anchor time-major shape "
                    f"{tuple(expected_shape)}, got {tuple(opponent_context_index.shape)}"
                )
            trunk_kwargs["opponent_context_index"] = opponent_context_index
        seat_hidden_state = self._prepare_seat_hidden_state(
            _batch_value(batch, "initial_hidden_state"),
            batch_size=batch_size,
            like=obs,
        )
        recurrent_flat, state_repr, observation_context, _values, _seat_hidden = cast(
            Any, forward_model
        ).forward_trunk_sequence_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            **trunk_kwargs,
        )
        return self._factorized_packed_candidate_log_probs(
            forward_model,
            recurrent_flat=recurrent_flat,
            obs_rows=obs.reshape(int(expected_shape[0] * expected_shape[1]), obs.shape[-1]),
            legal_actions=self._packed_legal_action_view(packed_legal),
            state_repr=state_repr,
            observation_context={} if observation_context is None else dict(observation_context),
            opponent_context_index=(None if opponent_context_index is None else opponent_context_index.reshape(-1)),
        )

    def _policy_anchor_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
        factorized_result: Any,
        forward_model: nn.Module,
        reset_before_step: Tensor | None,
    ) -> tuple[Tensor | None, dict[str, float]]:
        kl_coef = float(self.policy_anchor_coef)
        top_action_coef = float(self.policy_anchor_top_action_coef)
        if kl_coef == 0.0 and top_action_coef == 0.0:
            return None, {}
        if packed_legal is None or factorized_result is None:
            raise ValueError("policy_anchor_coef currently requires the factorized packed learner path")
        anchor_model = self._ensure_policy_anchor_model()
        if not self._should_use_factorized_legal_policy(anchor_model, packed_legal=packed_legal):
            raise ValueError("policy_anchor_coef requires a factorized structured anchor model")
        anchor_started = time.perf_counter()
        current_log_probs = self._factorized_candidate_log_probs_for_model(
            forward_model,
            batch,
            obs=obs,
            packed_legal=packed_legal,
            reset_before_step=reset_before_step,
        )
        with torch.no_grad():
            anchor_log_probs = self._factorized_candidate_log_probs_for_model(
                anchor_model,
                batch,
                obs=obs,
                packed_legal=packed_legal,
                reset_before_step=reset_before_step,
            )
        total_anchor_loss = current_log_probs.sum() * 0.0
        anchor_metrics: dict[str, float] = {}
        if kl_coef != 0.0:
            anchor_loss, anchor_metrics = packed_candidate_anchor_kl_loss(
                current_log_probs=current_log_probs,
                anchor_log_probs=anchor_log_probs,
                packed_offsets=packed_legal[1],
                row_shape=(int(obs.shape[0]), int(obs.shape[1])),
                loss_mask=loss_mask,
                temperature=float(self.policy_anchor_temperature),
            )
            total_anchor_loss = total_anchor_loss + (anchor_loss * kl_coef)
            anchor_metrics["policy_anchor_coef_active"] = kl_coef
            anchor_metrics["policy_anchor_temperature"] = float(self.policy_anchor_temperature)
        if top_action_coef != 0.0:
            top_action_loss, top_action_metrics = packed_candidate_anchor_top_action_loss(
                current_log_probs=current_log_probs,
                anchor_log_probs=anchor_log_probs,
                packed_offsets=packed_legal[1],
                row_shape=(int(obs.shape[0]), int(obs.shape[1])),
                loss_mask=loss_mask,
            )
            total_anchor_loss = total_anchor_loss + (top_action_loss * top_action_coef)
            anchor_metrics.update(top_action_metrics)
            anchor_metrics["policy_anchor_top_action_coef_active"] = top_action_coef
        self._record_timing_ms("learner_policy_anchor", time.perf_counter() - anchor_started)
        anchor_metrics["policy_anchor_weighted_loss"] = float(total_anchor_loss.detach().item())
        return total_anchor_loss, anchor_metrics


__all__ = ["ImpalaPolicyAnchorSupportMixin"]
