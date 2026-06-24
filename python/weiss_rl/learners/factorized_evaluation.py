"""Factorized policy evaluation helpers for the IMPALA learner."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.factorized_batch import factorized_batch_value as _batch_value
from weiss_rl.learners.factorized_public_teacher import ImpalaFactorizedPublicTeacherMixin


class ImpalaFactorizedEvaluationMixin(ImpalaFactorizedPublicTeacherMixin):
    """Evaluates the learner-side factorized policy path over time-major batches."""

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
