"""Public-heuristic teacher view for learner-side factorized policy distillation."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.factorized_batch import factorized_batch_value as _batch_value
from weiss_rl.learners.structured_legal_view import (
    PackedStructuredLegalView as _PackedStructuredLegalView,
)
from weiss_rl.learners.structured_legal_view import (
    packed_structured_legal_view as _packed_structured_legal_view,
)


def _attach_initial_hidden_context_gradient(
    forward_model: Any,
    seat_hidden_state: Tensor,
    opponent_context_index: Tensor | None,
) -> Tensor:
    if opponent_context_index is None:
        return seat_hidden_state
    if seat_hidden_state.ndim != 3:
        return seat_hidden_state
    context_fn = getattr(forward_model, "_opponent_context_hidden", None)
    if not callable(context_fn):
        return seat_hidden_state
    context_index = torch.as_tensor(opponent_context_index, device=seat_hidden_state.device, dtype=torch.long)
    if context_index.ndim == 2:
        initial_context_index = context_index[0]
    elif context_index.ndim == 1:
        initial_context_index = context_index
    else:
        return seat_hidden_state
    batch_size = int(seat_hidden_state.shape[0])
    if int(initial_context_index.numel()) != batch_size:
        return seat_hidden_state
    context = context_fn(
        batch_size=batch_size,
        device=seat_hidden_state.device,
        dtype=seat_hidden_state.dtype,
        opponent_policy_ids=None,
        opponent_context_indices=initial_context_index,
    )
    if context is None:
        return seat_hidden_state
    return seat_hidden_state + (context - context.detach()).unsqueeze(1)


class ImpalaFactorizedPublicTeacherMixin:
    """Builds factorized student/teacher logits for public-heuristic distillation."""

    def _factorized_public_heuristic_teacher_view(
        self: Any,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        score_public_target: bool = True,
        reattach_initial_hidden_context_gradient: bool = False,
    ) -> tuple[_PackedStructuredLegalView, Tensor | None] | tuple[None, None]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model for factorized public-heuristic distillation")
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not hasattr(forward_model, "forward_trunk_sequence_seat_aware"):
            return None, None
        if not self._has_factorized_packed_candidate_log_probs(forward_model):
            raise ValueError("factorized teacher candidate view requires factorized packed action log-probs")
        if score_public_target and not hasattr(forward_model, "score_packed_public_heuristic_candidates"):
            raise ValueError("factorized public-heuristic distillation requires public heuristic candidate scores")

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        total_rows = int(expected_shape[0] * expected_shape[1])
        active_rows = torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            return None, None

        acting_seat = self._prepare_acting_seat_batch(
            _batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            expected_shape=expected_shape,
        )
        if acting_seat is None:
            raise ValueError("factorized public-heuristic distillation requires acting seat information")

        flat_obs = obs.reshape(total_rows, obs.shape[-1])
        seat_hidden_state = self._prepare_seat_hidden_state(
            _batch_value(batch, "initial_hidden_state"),
            batch_size=batch_size,
            like=obs,
        )
        opponent_context_index = _batch_value(batch, "opponent_context_index")
        if opponent_context_index is not None:
            opponent_context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
            if tuple(opponent_context_index.shape) != tuple(expected_shape):
                raise ValueError(
                    "opponent_context_index must match factorized learner time-major shape "
                    f"{tuple(expected_shape)}, got {tuple(opponent_context_index.shape)}"
                )
            if reattach_initial_hidden_context_gradient:
                seat_hidden_state = _attach_initial_hidden_context_gradient(
                    forward_model,
                    seat_hidden_state,
                    opponent_context_index,
                )

        student_started = time.perf_counter()
        trunk_kwargs = {} if opponent_context_index is None else {"opponent_context_index": opponent_context_index}
        recurrent_flat, state_repr, observation_context, _values, _seat_hidden = (
            forward_model.forward_trunk_sequence_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                **trunk_kwargs,
            )
        )

        if int(active_rows.shape[0]) == total_rows:
            legal_actions_view = self._packed_legal_action_view(packed_legal)
            student_subset_logits = self._factorized_packed_candidate_log_probs(
                forward_model,
                recurrent_flat=recurrent_flat,
                obs_rows=flat_obs,
                legal_actions=legal_actions_view,
                state_repr=state_repr,
                observation_context=observation_context,
                opponent_context_index=(None if opponent_context_index is None else opponent_context_index.reshape(-1)),
            )
            self._record_timing_ms("learner_public_heuristic_student", time.perf_counter() - student_started)

            target_logits = None
            if score_public_target:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    target_logits = self._score_public_teacher_target_logits(
                        forward_model=forward_model,
                        obs_rows=flat_obs,
                        legal_actions=legal_actions_view,
                        observation_context=observation_context,
                        device=recurrent_flat.device,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)
            student_logits = student_subset_logits
        else:
            subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
            subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
            subset_obs = flat_obs.index_select(0, active_rows)
            subset_context = self._subset_observation_context_rows(
                observation_context,
                active_rows,
                row_count=total_rows,
            )
            student_subset_logits = self._factorized_packed_candidate_log_probs(
                forward_model,
                recurrent_flat=recurrent_flat.index_select(0, active_rows),
                obs_rows=subset_obs,
                legal_actions=subset_legal_actions,
                state_repr=state_repr.index_select(0, active_rows),
                observation_context=subset_context,
                opponent_context_index=(
                    None
                    if opponent_context_index is None
                    else opponent_context_index.reshape(-1).index_select(0, active_rows)
                ),
            )
            self._record_timing_ms("learner_public_heuristic_student", time.perf_counter() - student_started)

            target_subset_logits = None
            if score_public_target:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    target_subset_logits = self._score_public_teacher_target_logits(
                        forward_model=forward_model,
                        obs_rows=subset_obs,
                        legal_actions=subset_legal_actions,
                        observation_context=subset_context,
                        device=recurrent_flat.device,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)
            student_logits = self._scatter_packed_candidate_values(
                packed_legal,
                active_rows,
                student_subset_logits,
                fill_value=0.0,
            )
            target_logits = (
                None
                if target_subset_logits is None
                else self._scatter_packed_candidate_values(
                    packed_legal,
                    active_rows,
                    target_subset_logits,
                    fill_value=0.0,
                )
            )

        packed_view = _packed_structured_legal_view(
            logits=student_logits,
            packed_ids=packed_legal[0],
            packed_offsets=packed_legal[1],
            packed_meta=packed_legal[2],
        )
        assert packed_view is not None
        return packed_view, target_logits

    def _has_factorized_packed_candidate_log_probs(self: Any, forward_model: Any) -> bool:
        policy_head = getattr(forward_model, "policy_head", None)
        return callable(getattr(policy_head, "factorized_packed_action_log_probs", None))

    def _factorized_packed_candidate_log_probs(
        self: Any,
        forward_model: Any,
        *,
        recurrent_flat: Tensor,
        obs_rows: Tensor,
        legal_actions: Any,
        state_repr: Tensor,
        observation_context: dict[str, Tensor],
        opponent_context_index: Tensor | None = None,
    ) -> Tensor:
        contextual_scorer = getattr(forward_model, "_factorized_packed_action_log_probs_with_context", None)
        if callable(contextual_scorer):
            candidate_log_probs = contextual_scorer(
                recurrent_flat,
                obs=obs_rows,
                legal_actions=legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
                opponent_context_index=opponent_context_index,
            )
            return torch.as_tensor(candidate_log_probs, device=recurrent_flat.device)
        policy_head = getattr(forward_model, "policy_head", None)
        scorer = getattr(policy_head, "factorized_packed_action_log_probs", None)
        if not callable(scorer):
            raise ValueError("factorized teacher candidate view requires factorized packed action log-probs")
        candidate_log_probs = scorer(
            recurrent_flat,
            obs=obs_rows,
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
        )
        candidate_log_probs = torch.as_tensor(candidate_log_probs, device=recurrent_flat.device)
        apply_candidate_residual = getattr(
            forward_model,
            "_apply_opponent_context_packed_candidate_residual_to_log_probs",
            None,
        )
        if callable(apply_candidate_residual):
            candidate_log_probs = apply_candidate_residual(
                candidate_log_probs,
                legal_actions,
                state_repr,
                opponent_context_index,
                observation_context=observation_context,
                scoring_mode="learner",
            )
        apply_context_bias = getattr(forward_model, "_apply_opponent_context_packed_action_bias_to_log_probs", None)
        if callable(apply_context_bias):
            candidate_log_probs = apply_context_bias(
                candidate_log_probs,
                legal_actions,
                opponent_context_index,
            )
        return candidate_log_probs


__all__ = ["ImpalaFactorizedPublicTeacherMixin"]
