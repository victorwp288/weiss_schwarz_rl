"""IMPALA learner forward-pass support."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.forward_time_major import ForwardTimeMajorResult
from weiss_rl.learners.forward_time_major import forward_time_major as learner_forward_time_major


class ImpalaForwardSupportMixin:
    def _forward_time_major(
        self: Any,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: LegalActionBatch | None = None,
        policy_train_mask: Tensor | None = None,
        reset_before_step: Tensor | None = None,
        opponent_context_index: Any = None,
    ) -> ForwardTimeMajorResult:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        return learner_forward_time_major(
            model=self.model,
            compiled_model=self.compiled_model,
            obs=obs,
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            actor=actor,
            legal_actions=legal_actions,
            policy_train_mask=policy_train_mask,
            reset_before_step=reset_before_step,
            opponent_context_index=opponent_context_index,
            prepare_acting_seat_batch=self._prepare_acting_seat_batch,
            prepare_legacy_hidden_state=self._prepare_legacy_hidden_state,
            prepare_seat_hidden_state=self._prepare_seat_hidden_state,
            slice_packed_legal_rows_with_meta=self._slice_packed_legal_rows_with_meta,
            packed_legal_action_view=self._packed_legal_action_view,
            subset_observation_context_rows=self._subset_observation_context_rows,
            scatter_packed_candidate_values=self._scatter_packed_candidate_values,
            record_timing_ms=self._record_timing_ms,
            active_timing_metrics=self._active_timing_metrics,
        )


__all__ = ["ImpalaForwardSupportMixin"]
