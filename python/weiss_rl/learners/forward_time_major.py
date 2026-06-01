"""Time-major model-forward dispatch for IMPALA-style learners."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

import numpy as np
import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch


@dataclass(frozen=True, slots=True)
class ForwardTimeMajorResult:
    values: Tensor
    logits: Tensor | None = None
    packed_logits: Tensor | None = None
    observation_context: Mapping[str, Tensor] | None = None

    def __iter__(self):
        yield self.logits if self.logits is not None else self.packed_logits
        yield self.values


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if keyword in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def _add_opponent_context_if_supported(callable_obj: Any, kwargs: dict[str, Any], context_index: Tensor | None) -> None:
    if context_index is not None and _call_accepts_keyword(callable_obj, "opponent_context_index"):
        kwargs["opponent_context_index"] = context_index


def time_step_legal_actions(
    legal_actions: LegalActionBatch | None, *, step_index: int, batch_size: int
) -> LegalActionBatch | None:
    if legal_actions is None:
        return None
    if legal_actions.mask is not None:
        mask = np.asarray(legal_actions.mask, dtype=np.bool_)
        if mask.ndim != 3 or mask.shape[1] != batch_size:
            raise ValueError("legal mask must have shape (time, batch, action) matching the learner batch")
        if step_index < 0 or step_index >= mask.shape[0]:
            raise ValueError("step_index is outside the legal mask time dimension")
        return LegalActionBatch.from_mask(np.expand_dims(mask[step_index], axis=0))
    if legal_actions.ids is None or legal_actions.offsets is None:
        return None
    ids = np.asarray(legal_actions.ids, dtype=np.uint32)
    offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
    row_start = int(step_index * batch_size)
    row_stop = int(row_start + batch_size)
    if offsets.ndim != 1 or row_stop + 1 > offsets.size:
        raise ValueError("packed legal offsets must match the learner batch shape")
    start = int(offsets[row_start])
    stop = int(offsets[row_stop])
    slice_ids = np.array(ids[start:stop], copy=True)
    slice_offsets = np.array(offsets[row_start : row_stop + 1] - offsets[row_start], copy=True)
    return LegalActionBatch.from_packed(slice_ids, slice_offsets)


def forward_time_major(
    *,
    model: Any,
    compiled_model: Any | None,
    obs: Tensor,
    initial_hidden_state: Any = None,
    to_play_seat: Any = None,
    actor: Any = None,
    legal_actions: LegalActionBatch | None = None,
    policy_train_mask: Tensor | None = None,
    reset_before_step: Tensor | None = None,
    opponent_context_index: Any = None,
    prepare_acting_seat_batch: Callable[..., Tensor | None],
    prepare_legacy_hidden_state: Callable[..., Tensor | None],
    prepare_seat_hidden_state: Callable[..., Tensor | None],
    slice_packed_legal_rows_with_meta: Callable[
        [tuple[Tensor, Tensor, Tensor | None], Tensor], tuple[Tensor, Tensor, Tensor | None]
    ],
    packed_legal_action_view: Callable[[tuple[Tensor, Tensor, Tensor | None]], Any],
    subset_observation_context_rows: Callable[..., dict[str, Tensor]],
    scatter_packed_candidate_values: Callable[..., Tensor],
    record_timing_ms: Callable[[str, float], None],
    active_timing_metrics: dict[str, float] | None,
) -> ForwardTimeMajorResult:
    forward_model = compiled_model if compiled_model is not None else model
    if obs.ndim != 3:
        raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")

    expected_shape = obs.shape[:2]
    batch_size = int(obs.shape[1])
    structured_legal_actions = bool(getattr(forward_model, "supports_legal_candidate_scoring", False))
    acting_seat = prepare_acting_seat_batch(to_play_seat, actor=actor, expected_shape=expected_shape)
    if structured_legal_actions and legal_actions is not None:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("structured learner updates require packed legal_actions ids/offsets")
        if legal_actions.meta is None:
            raise ValueError("structured learner updates require packed legal_actions metadata")
    context_index = None
    if opponent_context_index is not None:
        context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
        if context_index.ndim != 2 or tuple(context_index.shape) != tuple(expected_shape):
            raise ValueError("opponent_context_index must be 2D (time, batch) with the same leading dimensions as obs")
    sequence_started = time.perf_counter()
    if (
        acting_seat is not None
        and structured_legal_actions
        and legal_actions is not None
        and hasattr(forward_model, "forward_trunk_sequence_seat_aware")
    ):
        legal_action_ids = legal_actions.ids
        legal_action_offsets = legal_actions.offsets
        legal_action_meta = legal_actions.meta
        assert legal_action_ids is not None
        assert legal_action_offsets is not None
        assert legal_action_meta is not None
        trunk_started = time.perf_counter()
        trunk_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
        _add_opponent_context_if_supported(
            forward_model.forward_trunk_sequence_seat_aware,
            trunk_kwargs,
            context_index,
        )
        recurrent_flat, state_repr, observation_context, values, _next_hidden = (
            forward_model.forward_trunk_sequence_seat_aware(
                obs,
                acting_seat,
                prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
                **trunk_kwargs,
            )
        )
        record_timing_ms("learner_trunk", time.perf_counter() - trunk_started)
        restricted_rows = (
            policy_train_mask.reshape(-1).to(device=recurrent_flat.device, dtype=torch.bool)
            if policy_train_mask is not None
            else None
        )
        active_rows = None if restricted_rows is None else torch.nonzero(restricted_rows, as_tuple=False).squeeze(1)
        packed_logits: Tensor
        scorer_started = time.perf_counter()
        if active_rows is None or int(active_rows.shape[0]) == int(recurrent_flat.shape[0]):
            scorer_kwargs: dict[str, Any] = {
                "state_repr": state_repr,
                "observation_context": observation_context,
                "scoring_mode": "learner",
            }
            _add_opponent_context_if_supported(
                forward_model.score_packed_legal_candidates,
                scorer_kwargs,
                None if context_index is None else context_index.reshape(-1),
            )
            packed_logits = forward_model.score_packed_legal_candidates(
                recurrent_flat,
                obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
                legal_actions,
                **scorer_kwargs,
            )
        else:
            packed_legal = (
                torch.as_tensor(legal_action_ids, device=recurrent_flat.device, dtype=torch.long),
                torch.as_tensor(legal_action_offsets, device=recurrent_flat.device, dtype=torch.long),
                torch.as_tensor(legal_action_meta, device=recurrent_flat.device, dtype=torch.long),
            )
            subset_packed_legal = slice_packed_legal_rows_with_meta(packed_legal, active_rows)
            subset_legal_actions = packed_legal_action_view(subset_packed_legal)
            if active_rows.numel() == 0:
                subset_logits = recurrent_flat.new_zeros((0,))
            else:
                subset_scorer_kwargs: dict[str, Any] = {
                    "state_repr": state_repr.index_select(0, active_rows),
                    "observation_context": subset_observation_context_rows(
                        observation_context,
                        active_rows,
                        row_count=int(recurrent_flat.shape[0]),
                    ),
                    "scoring_mode": "learner",
                }
                _add_opponent_context_if_supported(
                    forward_model.score_packed_legal_candidates,
                    subset_scorer_kwargs,
                    None if context_index is None else context_index.reshape(-1).index_select(0, active_rows),
                )
                subset_logits = torch.as_tensor(
                    forward_model.score_packed_legal_candidates(
                        recurrent_flat.index_select(0, active_rows),
                        obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]).index_select(0, active_rows),
                        subset_legal_actions,
                        **subset_scorer_kwargs,
                    ),
                    device=recurrent_flat.device,
                )
            packed_logits = scatter_packed_candidate_values(
                packed_legal,
                active_rows,
                subset_logits,
                fill_value=0.0,
            )
        record_timing_ms("learner_packed_scorer", time.perf_counter() - scorer_started)
        record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
        packed_rows = int(legal_action_offsets.shape[0] - 1)
        packed_candidates = int(legal_action_ids.shape[0])
        metrics = {
            "packed_candidate_count": float(packed_candidates),
            "packed_candidate_rows": float(packed_rows),
            "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
        }
        if active_rows is not None:
            active_rows_count = int(active_rows.shape[0])
            if active_rows_count == packed_rows:
                active_candidates = packed_candidates
            else:
                subset_offsets = subset_packed_legal[1]
                active_candidates = int(subset_offsets[-1].item()) if subset_offsets.numel() > 0 else 0
            metrics.update(
                {
                    "packed_candidate_train_count": float(active_candidates),
                    "packed_candidate_train_rows": float(active_rows_count),
                }
            )
        if active_timing_metrics is not None:
            active_timing_metrics.update(metrics)
        return ForwardTimeMajorResult(
            packed_logits=torch.as_tensor(packed_logits),
            values=torch.as_tensor(values),
            observation_context=observation_context,
        )
    if acting_seat is not None and hasattr(forward_model, "forward_sequence_seat_aware"):
        sequence_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
        _add_opponent_context_if_supported(
            forward_model.forward_sequence_seat_aware,
            sequence_kwargs,
            context_index,
        )
        logits, values, _next_hidden = forward_model.forward_sequence_seat_aware(
            obs,
            acting_seat,
            prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
            legal_actions=legal_actions if structured_legal_actions else None,
            **sequence_kwargs,
        )
        record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
        metrics = {}
        if structured_legal_actions and legal_actions is not None and legal_actions.offsets is not None:
            packed_rows = int(legal_actions.offsets.shape[0] - 1)
            packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
            metrics = {
                "packed_candidate_count": float(packed_candidates),
                "packed_candidate_rows": float(packed_rows),
                "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
            }
        if active_timing_metrics is not None:
            active_timing_metrics.update(metrics)
        return ForwardTimeMajorResult(
            logits=torch.as_tensor(logits),
            values=torch.as_tensor(values),
        )
    logits_steps: list[Tensor] = []
    value_steps: list[Tensor] = []

    if acting_seat is None:
        hidden_state = prepare_legacy_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
        for step_index, step_obs in enumerate(obs.unbind(dim=0)):
            if reset_before_step is not None and hidden_state is not None:
                step_reset = reset_before_step[step_index].to(device=hidden_state.device, dtype=torch.bool)
                if bool(step_reset.any().item()):
                    hidden_state = hidden_state.clone()
                    hidden_state[step_reset] = torch.zeros_like(hidden_state[step_reset])
            step_legal_actions = (
                time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
                if structured_legal_actions
                else None
            )
            if step_legal_actions is None:
                step_logits, step_value, hidden_state = forward_model(step_obs, hidden_state)
            else:
                step_logits, step_value, hidden_state = forward_model(
                    step_obs,
                    hidden_state,
                    legal_actions=step_legal_actions,
                )
            logits_steps.append(torch.as_tensor(step_logits))
            value_steps.append(torch.as_tensor(step_value))
            if hidden_state is not None:
                hidden_state = torch.as_tensor(hidden_state)
        return ForwardTimeMajorResult(
            logits=torch.stack(logits_steps, dim=0),
            values=torch.stack(value_steps, dim=0),
        )

    seat_hidden_state = prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
    for step_index, (step_obs, step_seat) in enumerate(zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)):
        if reset_before_step is not None and seat_hidden_state is not None:
            step_reset = reset_before_step[step_index].to(device=seat_hidden_state.device, dtype=torch.bool)
            if bool(step_reset.any().item()):
                seat_hidden_state = seat_hidden_state.clone()
                seat_hidden_state[step_reset] = torch.zeros_like(seat_hidden_state[step_reset])
        step_legal_actions = (
            time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
            if structured_legal_actions
            else None
        )
        if step_legal_actions is None:
            step_logits, step_value, seat_hidden_state = forward_model.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden_state,
            )
        else:
            step_logits, step_value, seat_hidden_state = forward_model.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden_state,
                legal_actions=step_legal_actions,
            )
        logits_steps.append(torch.as_tensor(step_logits))
        value_steps.append(torch.as_tensor(step_value))
        if seat_hidden_state is not None:
            seat_hidden_state = torch.as_tensor(seat_hidden_state)
    record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
    return ForwardTimeMajorResult(
        logits=torch.stack(logits_steps, dim=0),
        values=torch.stack(value_steps, dim=0),
    )
