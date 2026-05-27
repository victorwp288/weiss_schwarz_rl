"""Factorized policy evaluation helpers for the IMPALA learner."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView as _PackedStructuredLegalView,
)
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


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


class ImpalaFactorizedEvaluationMixin:
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
                    target_logits = self._score_public_heuristic_target_logits(
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
                    target_subset_logits = self._score_public_heuristic_target_logits(
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

    def _should_use_factorized_legal_policy(
        self: Any, forward_model: Any, *, packed_legal: tuple[Tensor, Tensor, Tensor | None] | None
    ) -> bool:
        return bool(
            packed_legal is not None
            and getattr(forward_model, "supports_factorized_legal_policy", False)
            and hasattr(forward_model, "evaluate_factorized_sequence_packed_seat_aware")
        )

    def _evaluate_factorized_time_major(
        self: Any,
        batch: Any,
        *,
        obs: Tensor,
        actions: Tensor | None,
        extra_active_mask: Tensor | None = None,
    ) -> tuple[Any, tuple[Tensor, Tensor, Tensor | None]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to evaluate factorized legal policies")
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        expected_shape = obs.shape[:2]
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if packed_legal is None:
            raise ValueError("factorized learner updates require packed legal actions")
        if not self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            raise ValueError("factorized learner updates require a factorized structured policy model")
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            _batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            expected_shape=expected_shape,
        )
        if acting_seat is None:
            raise ValueError("factorized learner updates require acting seat information")
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        if extra_active_mask is not None:
            extra_active = extra_active_mask.to(device=obs.device, dtype=torch.bool)
            if tuple(extra_active.shape) != tuple(expected_shape):
                raise ValueError(
                    "extra_active_mask must match factorized learner time-major shape "
                    f"{tuple(expected_shape)}, got {tuple(extra_active.shape)}"
                )
            if loss_mask is None:
                loss_mask = extra_active.to(dtype=obs.dtype)
            else:
                loss_mask = torch.logical_or(loss_mask > 0.0, extra_active).to(dtype=loss_mask.dtype)
        active_rows = (
            None if loss_mask is None else torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        )
        same_family_reference_actions = None
        same_family_reference_families = None
        if (
            float(self.teacher_same_family_action_coef) != 0.0
            or float(self.teacher_action_coef) != 0.0
            or float(getattr(self, "teacher_hand_coef", 0.0)) != 0.0
        ):
            raw_teacher_action = _batch_value(batch, "teacher_action")
            raw_teacher_family = _batch_value(batch, "teacher_family")
            if raw_teacher_action is not None and raw_teacher_family is not None:
                same_family_reference_actions = self._tensor_on_model_device(raw_teacher_action, dtype=torch.long)
                same_family_reference_families = self._tensor_on_model_device(raw_teacher_family, dtype=torch.long)
                if same_family_reference_actions.shape != expected_shape:
                    raise ValueError(
                        "teacher_action must match factorized learner time-major shape "
                        f"{tuple(expected_shape)}, got {tuple(same_family_reference_actions.shape)}"
                    )
                if same_family_reference_families.shape != expected_shape:
                    raise ValueError(
                        "teacher_family must match factorized learner time-major shape "
                        f"{tuple(expected_shape)}, got {tuple(same_family_reference_families.shape)}"
                    )
        factorized_started = time.perf_counter()
        seat_hidden_state = self._prepare_seat_hidden_state(
            _batch_value(batch, "initial_hidden_state"),
            batch_size=batch_size,
            like=obs,
        )
        reset_before_step = self._optional_time_major_loss_mask(
            _batch_value(batch, "reset_before_step"),
            expected_shape=expected_shape,
            like=obs[..., 0],
        )
        if reset_before_step is not None:
            reset_before_step = reset_before_step.to(dtype=torch.bool)
        opponent_context_index = _batch_value(batch, "opponent_context_index")
        if opponent_context_index is not None:
            opponent_context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
            if tuple(opponent_context_index.shape) != tuple(expected_shape):
                raise ValueError(
                    "opponent_context_index must match factorized learner time-major shape "
                    f"{tuple(expected_shape)}, got {tuple(opponent_context_index.shape)}"
                )
        total_rows = int(expected_shape[0] * expected_shape[1])
        if active_rows is None or active_rows.numel() == 0 or int(active_rows.shape[0]) == total_rows:
            sequence_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
            if opponent_context_index is not None:
                sequence_kwargs["opponent_context_index"] = opponent_context_index
            result = forward_model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                legal_actions=self._packed_legal_action_view(packed_legal),
                actions=actions,
                same_family_reference_actions=same_family_reference_actions,
                same_family_reference_families=same_family_reference_families,
                **sequence_kwargs,
            )
        else:
            trunk_kwargs = {} if reset_before_step is None else {"reset_before_step": reset_before_step}
            if opponent_context_index is not None:
                trunk_kwargs["opponent_context_index"] = opponent_context_index
            recurrent_flat, state_repr, observation_context, values, _seat_hidden = (
                forward_model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                    **trunk_kwargs,
                )
            )
            policy_head = forward_model.policy_head
            full_plan = policy_head._build_factorized_legality_plan(
                self._packed_legal_action_view(packed_legal),
                device=state_repr.device,
            )
            family_log_probs_full = policy_head._family_log_probs(
                state_repr,
                full_plan.family_mask,
                full_plan.family_candidate_counts,
            )
            subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
            subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
            flat_obs = obs.reshape(total_rows, obs.shape[-1])
            subset_result = policy_head.evaluate_factorized_packed(
                recurrent_flat.index_select(0, active_rows),
                obs=flat_obs.index_select(0, active_rows),
                legal_actions=subset_legal_actions,
                actions=None if actions is None else actions.reshape(-1).index_select(0, active_rows),
                same_family_reference_actions=(
                    None
                    if same_family_reference_actions is None
                    else same_family_reference_actions.reshape(-1).index_select(0, active_rows)
                ),
                same_family_reference_families=(
                    None
                    if same_family_reference_families is None
                    else same_family_reference_families.reshape(-1).index_select(0, active_rows)
                ),
                observation_context=self._subset_observation_context_rows(
                    observation_context,
                    active_rows,
                    row_count=total_rows,
                ),
                state_repr=state_repr.index_select(0, active_rows),
            )
            contextual_stats_fn = getattr(forward_model, "_factorized_contextual_packed_policy_stats", None)
            if callable(contextual_stats_fn):
                subset_context_index = (
                    None
                    if opponent_context_index is None
                    else opponent_context_index.reshape(-1).index_select(0, active_rows)
                )
                subset_contextual_stats = contextual_stats_fn(
                    recurrent_flat.index_select(0, active_rows),
                    obs=flat_obs.index_select(0, active_rows),
                    legal_actions=subset_legal_actions,
                    actions=None if actions is None else actions.reshape(-1).index_select(0, active_rows),
                    state_repr=state_repr.index_select(0, active_rows),
                    observation_context=self._subset_observation_context_rows(
                        observation_context,
                        active_rows,
                        row_count=total_rows,
                    ),
                    opponent_context_index=subset_context_index,
                )
                if subset_contextual_stats is not None:
                    subset_action_logp, subset_entropy, subset_top_action_ids = subset_contextual_stats
                    subset_result = replace(
                        subset_result,
                        action_logp=subset_action_logp,
                        entropy=subset_entropy,
                        top_action_ids=subset_top_action_ids,
                    )

            def _scatter_rows(values_subset: Tensor | None, *, fill_value: float = 0.0) -> Tensor | None:
                if values_subset is None:
                    return None
                full = values_subset.new_full((total_rows, *values_subset.shape[1:]), fill_value)
                full.index_copy_(0, active_rows, values_subset)
                return full

            def _scatter_log_probs(values_subset: Tensor | None) -> Tensor | None:
                if values_subset is None:
                    return None
                scattered = _scatter_rows(values_subset, fill_value=-torch.inf)
                assert scattered is not None
                return scattered.reshape(expected_shape[0], expected_shape[1], values_subset.shape[-1])

            def _scatter_time_major(values_subset: Tensor | None, *, fill_value: float) -> Tensor | None:
                if values_subset is None:
                    return None
                scattered = _scatter_rows(values_subset, fill_value=fill_value)
                assert scattered is not None
                return scattered.reshape(expected_shape)

            subset_top_action_ids = subset_result.top_action_ids
            subset_same_family_action_logp = subset_result.same_family_action_logp
            subset_same_family_top_action_ids = subset_result.same_family_top_action_ids
            subset_same_family_arg0_logp = getattr(subset_result, "same_family_arg0_logp", None)
            subset_same_family_top_arg0 = getattr(subset_result, "same_family_top_arg0", None)
            result = SimpleNamespace(
                values=values,
                action_logp=_scatter_rows(subset_result.action_logp),
                entropy=_scatter_rows(subset_result.entropy),
                family_log_probs=family_log_probs_full.reshape(
                    expected_shape[0], expected_shape[1], family_log_probs_full.shape[-1]
                ),
                play_slot_log_probs=_scatter_log_probs(subset_result.play_slot_log_probs),
                move_slot_log_probs=_scatter_log_probs(subset_result.move_slot_log_probs),
                attack_slot_log_probs=_scatter_log_probs(subset_result.attack_slot_log_probs),
                attack_type_log_probs=_scatter_log_probs(subset_result.attack_type_log_probs),
                top_action_ids=_scatter_time_major(subset_top_action_ids, fill_value=-1),
                same_family_action_logp=_scatter_time_major(
                    subset_same_family_action_logp,
                    fill_value=-torch.inf,
                ),
                same_family_top_action_ids=_scatter_time_major(
                    subset_same_family_top_action_ids,
                    fill_value=-1,
                ),
                same_family_arg0_logp=_scatter_time_major(
                    subset_same_family_arg0_logp,
                    fill_value=-torch.inf,
                ),
                same_family_top_arg0=_scatter_time_major(
                    subset_same_family_top_arg0,
                    fill_value=-1,
                ),
            )
            if result.action_logp is not None:
                result.action_logp = result.action_logp.reshape(expected_shape)
            if result.entropy is not None:
                result.entropy = result.entropy.reshape(expected_shape)
        elapsed = time.perf_counter() - factorized_started
        self._record_timing_ms("learner_forward_time_major", elapsed)
        self._record_timing_ms("learner_factorized_policy", elapsed)
        packed_rows = int(packed_legal[1].shape[0] - 1)
        packed_candidates = int(packed_legal[0].shape[0])
        metrics = {
            "packed_candidate_count": float(packed_candidates),
            "packed_candidate_rows": float(packed_rows),
            "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
        }
        if self._active_timing_metrics is not None:
            self._active_timing_metrics.update(metrics)
        return result, packed_legal
