from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

import weiss_rl.learners.impala_auxiliary_update as impala_auxiliary_update
import weiss_rl.learners.impala_loss_context_stage as impala_loss_context_stage
import weiss_rl.learners.impala_loss_metrics_stage as impala_loss_metrics_stage
import weiss_rl.learners.impala_loss_teacher_stage as impala_loss_teacher_stage
import weiss_rl.learners.impala_loss_teacher_targets_stage as impala_loss_teacher_targets_stage
import weiss_rl.learners.impala_loss_vtrace_stage as impala_loss_vtrace_stage
import weiss_rl.learners.impala_normal_update as impala_normal_update
import weiss_rl.learners.impala_paired_outcome_update as impala_paired_outcome_update
import weiss_rl.learners.impala_paired_swing_update as impala_paired_swing_update
import weiss_rl.learners.impala_teacher_auxiliary_call as impala_teacher_auxiliary_call
import weiss_rl.learners.impala_update_training_step as impala_update_training_step
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.action_logp import (
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
)
from weiss_rl.learners.impala_action_reductions import resolve_impala_action_reductions
from weiss_rl.learners.impala_batch_support import ImpalaBatchSupportMixin
from weiss_rl.learners.impala_fault_support import ImpalaFaultSupportMixin
from weiss_rl.learners.impala_forward_support import ImpalaForwardSupportMixin
from weiss_rl.learners.impala_learner import (
    ImpalaLearner,
    _chosen_action_outcome_metrics,
    _masked_action_logp_and_entropy,
    _packed_structured_legal_view,
    compute_structured_teacher_auxiliary_metrics,
    summarize_structured_policy_metrics,
)
from weiss_rl.learners.impala_logging_support import ImpalaLoggingSupportMixin
from weiss_rl.learners.impala_loss_assembly import assemble_impala_loss_inputs
from weiss_rl.learners.impala_loss_batch_inputs import ImpalaLossBatchInputs, resolve_impala_loss_batch_inputs
from weiss_rl.learners.impala_loss_context_stage import finalize_impala_loss_context_stage
from weiss_rl.learners.impala_loss_core import (
    attach_resolved_vtrace_context,
    compute_impala_loss_core,
    resolve_impala_value_loss_mask,
    resolve_impala_vtrace_clip_config,
)
from weiss_rl.learners.impala_loss_finalization import (
    apply_impala_teacher_auxiliary,
    finalize_impala_loss_context,
)
from weiss_rl.learners.impala_loss_forward_context import build_impala_forward_context
from weiss_rl.learners.impala_loss_inputs import (
    prepare_impala_loss_inputs,
    resolve_impala_loss_forward_flags,
    resolve_impala_loss_masks,
)
from weiss_rl.learners.impala_loss_legal_mask import resolve_impala_dense_legal_mask
from weiss_rl.learners.impala_loss_masks import (
    ImpalaLossForwardFlags,
    ImpalaLossMasks,
)
from weiss_rl.learners.impala_loss_masks import (
    resolve_impala_loss_masks as resolve_impala_loss_masks_stage,
)
from weiss_rl.learners.impala_loss_metrics import (
    build_impala_loss_metrics,
)
from weiss_rl.learners.impala_loss_metrics import (
    chosen_action_outcome_metrics as chosen_action_outcome_metrics_impl,
)
from weiss_rl.learners.impala_loss_metrics_stage import assemble_impala_loss_core_metrics
from weiss_rl.learners.impala_loss_objective_stage import compute_impala_objective_stage
from weiss_rl.learners.impala_loss_pipeline import (
    compute_impala_loss_and_metrics_with_context,
    resolve_impala_loss_action_reductions,
)
from weiss_rl.learners.impala_loss_policy_anchor_stage import apply_impala_policy_anchor_stage
from weiss_rl.learners.impala_loss_policy_forward import ImpalaPolicyForwardResult, evaluate_impala_policy_forward
from weiss_rl.learners.impala_loss_teacher_stage import apply_impala_teacher_auxiliary_stage
from weiss_rl.learners.impala_loss_teacher_targets_stage import prepare_impala_loss_teacher_target_inputs
from weiss_rl.learners.impala_loss_vtrace_stage import compute_impala_vtrace_stage
from weiss_rl.learners.impala_metrics_assembly import (
    ImpalaMetricAssemblyRequest,
    assemble_impala_loss_metrics,
)
from weiss_rl.learners.impala_objective_loss import compute_impala_objective_losses
from weiss_rl.learners.impala_optimizer_step import run_impala_optimizer_step
from weiss_rl.learners.impala_paired_auxiliary_batch import resolve_paired_auxiliary_batch_inputs
from weiss_rl.learners.impala_paired_outcome_auxiliary import ImpalaPairedOutcomeAuxiliaryMixin
from weiss_rl.learners.impala_paired_outcome_candidates import (
    PairedOutcomeCandidateLogps,
    compute_paired_outcome_candidate_logps,
)
from weiss_rl.learners.impala_paired_outcome_outputs import (
    build_paired_outcome_preference_context,
    build_paired_outcome_preference_metrics,
)
from weiss_rl.learners.impala_paired_swing_auxiliary import ImpalaPairedSwingAuxiliaryMixin
from weiss_rl.learners.impala_paired_swing_candidates import compute_paired_swing_candidate_view
from weiss_rl.learners.impala_paired_swing_outputs import build_paired_swing_auxiliary_metrics
from weiss_rl.learners.impala_policy_anchor_support import ImpalaPolicyAnchorSupportMixin
from weiss_rl.learners.impala_public_heuristic_support import ImpalaPublicHeuristicSupportMixin
from weiss_rl.learners.impala_structured_summary import (
    ImpalaStructuredSummaryRequest,
    compute_impala_structured_policy_summary,
)
from weiss_rl.learners.impala_structured_teacher_auxiliary import ImpalaStructuredTeacherAuxiliaryMixin
from weiss_rl.learners.impala_teacher_auxiliary_request import (
    compute_impala_teacher_auxiliary,
    resolve_impala_teacher_auxiliary_coefficients,
    resolve_impala_teacher_auxiliary_factorized_inputs,
    resolve_impala_teacher_auxiliary_inputs,
    resolve_impala_teacher_auxiliary_labels,
    resolve_impala_teacher_auxiliary_packed_inputs,
)
from weiss_rl.learners.impala_teacher_target_inputs import (
    ImpalaTeacherTargetInputs,
    prepare_impala_teacher_target_inputs,
    resolve_impala_teacher_target_plan,
)
from weiss_rl.learners.impala_update_bookkeeping import (
    begin_impala_update_scope,
    finalize_impala_update_scope,
    set_impala_model_train_mode,
)
from weiss_rl.learners.impala_update_logging import log_impala_update_metrics_if_due
from weiss_rl.learners.impala_update_loop import (
    ScopedOptimizerUpdateSpec,
    run_scoped_impala_optimizer_update,
)
from weiss_rl.learners.impala_update_loss_stage import build_scoped_impala_loss
from weiss_rl.learners.impala_update_training_inputs import (
    has_impala_training_inputs,
    missing_impala_training_input_fields,
    resolve_impala_update_vtrace_result,
    summarize_precomputed_vtrace_update_metrics,
    validate_impala_training_inputs,
)
from weiss_rl.learners.impala_update_training_step import run_impala_training_optimizer_step
from weiss_rl.learners.impala_vtrace_targets import resolve_impala_vtrace_targets
from weiss_rl.learners.structured_auxiliary import structured_catalog_metadata
from weiss_rl.learners.structured_teacher_auxiliary import (
    compute_structured_teacher_auxiliary_metrics as compute_structured_teacher_auxiliary_metrics_impl,
)
from weiss_rl.learners.structured_teacher_auxiliary import (
    resolve_structured_teacher_branch,
    resolve_structured_teacher_dispatch,
    resolve_structured_teacher_required_labels,
    resolve_structured_teacher_zero_context,
)
from weiss_rl.learners.structured_teacher_common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher_factorized_actions import compute_factorized_teacher_action_supervision
from weiss_rl.learners.structured_teacher_factorized_groups import compute_factorized_teacher_group_supervision
from weiss_rl.learners.structured_teacher_factorized_hand import compute_factorized_teacher_hand_supervision
from weiss_rl.learners.structured_teacher_packed import compute_packed_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher_packed_actions import compute_packed_teacher_action_supervision
from weiss_rl.learners.structured_teacher_packed_groups import compute_packed_teacher_group_supervision
from weiss_rl.learners.structured_teacher_packed_margins import compute_packed_teacher_margin_supervision
from weiss_rl.learners.structured_teacher_packed_public import compute_packed_teacher_public_supervision
from weiss_rl.learners.vtrace import VTraceTargets


class NaNLogitModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.bias.unsqueeze(0).expand(batch, -1).clone()
        logits[0, 0] = torch.nan
        values = torch.zeros(batch, dtype=obs.dtype, device=obs.device)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


def test_impala_learner_reexports_structured_teacher_auxiliary_metrics() -> None:
    assert compute_structured_teacher_auxiliary_metrics is compute_structured_teacher_auxiliary_metrics_impl


def test_impala_learner_reexports_chosen_action_outcome_metrics() -> None:
    assert _chosen_action_outcome_metrics is chosen_action_outcome_metrics_impl


def test_impala_learner_uses_canonical_paired_swing_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPairedSwingAuxiliaryMixin)


def test_impala_learner_uses_canonical_paired_outcome_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPairedOutcomeAuxiliaryMixin)


def test_impala_learner_uses_canonical_structured_teacher_auxiliary_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaStructuredTeacherAuxiliaryMixin)


def test_impala_learner_uses_canonical_fault_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaFaultSupportMixin)


def test_impala_learner_uses_canonical_public_heuristic_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPublicHeuristicSupportMixin)


def test_impala_learner_uses_canonical_batch_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaBatchSupportMixin)


def test_impala_learner_uses_canonical_forward_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaForwardSupportMixin)


def test_impala_learner_uses_canonical_logging_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaLoggingSupportMixin)


def test_impala_learner_uses_canonical_policy_anchor_support_mixin() -> None:
    assert isinstance(ImpalaLearner(), ImpalaPolicyAnchorSupportMixin)


class NaNGradientModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(action_dim))
        self.value_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.logit_bias.unsqueeze(0).expand(batch, -1).clone()
        logits.register_hook(lambda grad: torch.full_like(grad, torch.nan))
        values = self.value_bias.expand(batch)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class TinyPolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int = 2, action_dim: int = 3) -> None:
        super().__init__()
        self.policy = nn.Linear(observation_dim, action_dim)
        self.value = nn.Linear(observation_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        values = self.value(obs).squeeze(-1)
        next_hidden = torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class SeatAwareTinyPolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int = 2, action_dim: int = 2) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(observation_dim, 1)

    def _next_hidden(
        self,
        obs: torch.Tensor,
        seat_hidden_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if seat_hidden_state is None:
            return torch.zeros((int(obs.shape[0]), 2, 1), dtype=obs.dtype, device=obs.device)
        return seat_hidden_state + 1.0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat, legal_actions
        batch = int(obs.shape[0])
        logits = self.logit_bias.unsqueeze(0).expand(batch, -1)
        values = self.value(obs).squeeze(-1)
        return logits, values, self._next_hidden(obs, seat_hidden_state)

    def forward_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del legal_actions
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, next_hidden = self.forward_seat_aware(
            flat_obs,
            acting_seat.reshape(time_steps * batch_size),
            seat_hidden_state,
        )
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )

    def value_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del acting_seat, seat_hidden_state
        return self.value(obs).squeeze(-1)


class TinyStructuredTeacherModel(nn.Module):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__()
        self.action_catalog = action_catalog
        self.supports_legal_candidate_scoring = True
        self.policy = nn.Linear(observation_dim, action_catalog.action_space_size)
        self.value = nn.Linear(observation_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        if legal_actions is not None:
            legal_mask = torch.as_tensor(
                legal_actions.to_mask(expected_shape=(1, int(obs.shape[0])), action_space=logits.shape[-1])[0]
            )
            logits = torch.where(legal_mask.to(device=logits.device), logits, torch.full_like(logits, -1.0e9))
        values = self.value(obs).squeeze(-1)
        next_hidden = torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class ForwardProxyModel(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.forward_calls = 0

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        return self.base(obs, hidden_state)


class SequenceStructuredTeacherModel(TinyStructuredTeacherModel):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__(action_catalog, observation_dim=observation_dim)
        self.sequence_calls = 0
        self.step_calls = 0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.step_calls += 1
        hidden_state = (
            torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state
        )
        return self.forward(obs, hidden_state, legal_actions=legal_actions)

    def forward_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sequence_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, _next_hidden = self.forward(flat_obs, None, legal_actions=legal_actions)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )


class TrunkStructuredTeacherModel(TinyStructuredTeacherModel):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__(action_catalog, observation_dim=observation_dim)
        self.trunk_calls = 0
        self.scorer_calls = 0
        self.sequence_calls = 0
        self.scorer_row_count = 0
        self.scorer_candidate_count = 0

    def forward_trunk_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        self.trunk_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        values = self.value(flat_obs).squeeze(-1).reshape(time_steps, batch_size)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return flat_obs, flat_obs, {"flat_obs": flat_obs}, values, next_hidden

    def score_packed_legal_candidates(
        self,
        recurrent_outputs: torch.Tensor,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch,
        *,
        state_repr: torch.Tensor | None = None,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        self.scorer_calls += 1
        assert state_repr is not None
        assert observation_context is not None
        self.scorer_row_count = int(obs.shape[0])
        ids = torch.as_tensor(legal_actions.ids, device=obs.device, dtype=torch.long)
        self.scorer_candidate_count = int(ids.shape[0])
        offsets = torch.as_tensor(legal_actions.offsets, device=obs.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(int(lengths.shape[0]), device=obs.device, dtype=torch.long),
            lengths,
        )
        flat_logits = self.policy(obs)
        return flat_logits[row_indices, ids]

    def forward_sequence_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sequence_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, _next_hidden = self.forward(flat_obs, None, legal_actions=legal_actions)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )


class FactorizedStructuredTeacherModel(nn.Module):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.action_catalog = action_catalog
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.factorized_calls = 0
        self.trunk_calls = 0
        self.public_student_calls = 0
        self.public_target_calls = 0
        self.public_target_profiles: list[str] = []
        self.factorized_candidate_logp_calls = 0

    @property
    def policy_head(self) -> Any:
        return self

    def forward_trunk_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        del acting_seat, seat_hidden_state
        self.trunk_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        values = self.bias.expand(time_steps, batch_size)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return flat_obs, flat_obs, {"flat_obs": flat_obs}, values, next_hidden

    def score_packed_legal_candidates(
        self,
        recurrent_outputs: torch.Tensor,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        *,
        state_repr: torch.Tensor | None = None,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        del recurrent_outputs, state_repr, observation_context, scoring_mode
        self.public_student_calls += 1
        action_dim = int(self.action_catalog.action_space_size)
        flat_logits = torch.full((obs.shape[0], action_dim), -20.0, dtype=obs.dtype, device=obs.device)
        flat_logits[:, 0] = self.bias + 0.5
        flat_logits[:, 5] = self.bias + 4.0
        flat_logits[:, self.action_catalog.pass_action_id] = self.bias - 5.0
        ids = torch.as_tensor(legal_actions.ids, device=obs.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=obs.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(int(lengths.shape[0]), device=obs.device, dtype=torch.long),
            lengths,
        )
        return flat_logits[row_indices, ids]

    def score_packed_public_heuristic_candidates(
        self,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        *,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> torch.Tensor:
        del obs, observation_context
        self.public_target_calls += 1
        self.public_target_profiles.append(str(scoring_profile))
        ids = torch.as_tensor(legal_actions.ids, dtype=torch.long)
        logits = torch.full((int(ids.shape[0]),), -6.0, dtype=self.bias.dtype, device=self.bias.device)
        if str(scoring_profile) == "aggressive":
            logits = torch.where(ids == 5, self.bias + 4.5, logits)
            logits = torch.where(ids == 0, self.bias - 0.5, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 7.0, logits)
        elif str(scoring_profile) == "control":
            logits = torch.where(ids == 5, self.bias + 1.5, logits)
            logits = torch.where(ids == 0, self.bias + 2.0, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 5.0, logits)
        else:
            logits = torch.where(ids == 5, self.bias + 3.0, logits)
            logits = torch.where(ids == 0, self.bias + 0.0, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 6.0, logits)
        return logits

    def factorized_packed_action_log_probs(
        self,
        latent: torch.Tensor,
        *,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        observation_context: dict[str, torch.Tensor] | None = None,
        state_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del latent, obs, observation_context, state_repr
        self.factorized_candidate_logp_calls += 1
        ids = torch.as_tensor(legal_actions.ids, dtype=torch.long, device=self.bias.device)
        offsets = torch.as_tensor(legal_actions.offsets, dtype=torch.long, device=self.bias.device)
        raw_scores = torch.full((int(ids.shape[0]),), -6.0, dtype=self.bias.dtype, device=self.bias.device)
        raw_scores = torch.where(ids == 0, self.bias + 4.0, raw_scores)
        raw_scores = torch.where(ids == 11, self.bias + 4.0, raw_scores)
        raw_scores = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 7.0, raw_scores)
        row_count = int(offsets.shape[0] - 1)
        log_probs = torch.empty_like(raw_scores)
        for row_index in range(row_count):
            start = int(offsets[row_index].item())
            end = int(offsets[row_index + 1].item())
            log_probs[start:end] = torch.log_softmax(raw_scores[start:end], dim=0)
        return log_probs

    def evaluate_factorized_sequence_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | Any,
        actions: torch.Tensor | None = None,
        same_family_reference_actions: torch.Tensor | None = None,
        same_family_reference_families: torch.Tensor | None = None,
    ) -> Any:
        del legal_actions, acting_seat, seat_hidden_state, same_family_reference_families
        self.factorized_calls += 1
        time_steps, batch_size, _obs_dim = obs.shape
        family_index = {family.name: index for index, family in enumerate(self.action_catalog.families)}
        attack_type_index = {name: index for index, name in enumerate(self.action_catalog.attack_type_names)}
        family_count = len(self.action_catalog.families)
        max_stage = max(int(self.action_catalog.max_stage), 1)
        attack_slot_count = max(int(self.action_catalog.attack_slot_count), 1)
        attack_type_count = max(len(self.action_catalog.attack_type_names), 1)

        family_logits = torch.full(
            (time_steps, batch_size, family_count),
            -2.0,
            dtype=obs.dtype,
            device=obs.device,
        )
        family_logits[0, 0, family_index["main_play_character"]] = self.bias + 4.0
        if time_steps * batch_size > 1 and "attack" in family_index:
            family_logits.reshape(-1, family_count)[1, family_index["attack"]] = self.bias + 4.0
        family_log_probs = torch.log_softmax(family_logits, dim=-1)

        play_slot_logits = torch.zeros((time_steps, batch_size, max_stage), dtype=obs.dtype, device=obs.device)
        play_slot_logits[..., 0] = self.bias + 3.0
        play_slot_log_probs = torch.log_softmax(play_slot_logits, dim=-1)

        attack_slot_logits = torch.zeros(
            (time_steps, batch_size, attack_slot_count), dtype=obs.dtype, device=obs.device
        )
        attack_slot_logits[..., 0] = self.bias + 3.0
        attack_slot_log_probs = torch.log_softmax(attack_slot_logits, dim=-1)

        attack_type_logits = torch.zeros(
            (time_steps, batch_size, attack_type_count), dtype=obs.dtype, device=obs.device
        )
        attack_type_logits[..., attack_type_index.get("direct", 0)] = self.bias + 3.0
        attack_type_log_probs = torch.log_softmax(attack_type_logits, dim=-1)

        values = self.bias.expand(time_steps, batch_size)
        entropy = torch.ones((time_steps, batch_size), dtype=obs.dtype, device=obs.device) * (self.bias + 1.0)
        action_logp = None if actions is None else self.bias.expand(time_steps, batch_size) - 0.25
        top_action_ids = None
        same_family_action_logp = None
        same_family_top_action_ids = None
        same_family_arg0_logp = None
        same_family_top_arg0 = None
        if same_family_reference_actions is not None:
            same_family_action_logp = self.bias.expand(time_steps, batch_size) - 0.1
            same_family_top_action_ids = same_family_reference_actions.to(device=obs.device, dtype=torch.long)
            top_action_ids = same_family_reference_actions.to(device=obs.device, dtype=torch.long)
            flat_top_arg0 = []
            for action_id in same_family_reference_actions.reshape(-1).tolist():
                decoded = self.action_catalog.decode(int(action_id))
                flat_top_arg0.append(-1 if decoded.hand_index is None else int(decoded.hand_index))
            same_family_top_arg0 = torch.as_tensor(flat_top_arg0, dtype=torch.long, device=obs.device).reshape(
                time_steps,
                batch_size,
            )
            same_family_arg0_logp = self.bias.expand(time_steps, batch_size) - 0.05
        return SimpleNamespace(
            values=values,
            action_logp=action_logp,
            entropy=entropy,
            family_log_probs=family_log_probs,
            play_slot_log_probs=play_slot_log_probs,
            move_source_log_probs=play_slot_log_probs,
            move_slot_log_probs=play_slot_log_probs,
            attack_slot_log_probs=attack_slot_log_probs,
            attack_type_log_probs=attack_type_log_probs,
            top_action_ids=top_action_ids,
            same_family_action_logp=same_family_action_logp,
            same_family_top_action_ids=same_family_top_action_ids,
            same_family_arg0_logp=same_family_arg0_logp,
            same_family_top_arg0=same_family_top_arg0,
        )


class _ScaledLoss:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss

    def backward(self) -> None:
        self.loss.backward()


class FakeGradScaler:
    def __init__(self, *, scale: float = 8.0, overflow: bool = False) -> None:
        self.scale_value = float(scale)
        self.overflow = overflow

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss: torch.Tensor) -> _ScaledLoss:
        return _ScaledLoss(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.overflow:
            optimizer.step()

    def update(self, new_scale: float | None = None) -> None:
        if new_scale is not None:
            self.scale_value = float(new_scale)
        elif self.overflow:
            self.scale_value *= 0.5


def test_run_impala_optimizer_step_reports_no_grad_without_stepping() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=torch.tensor(2.0),
        base_metrics={"loss": 2.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["loss"] == pytest.approx(2.0)
    assert metrics["optimizer_no_grad"] == pytest.approx(1.0)
    assert metrics["amp_grad_overflow"] == pytest.approx(0.0)
    assert metrics["loss_scale"] == pytest.approx(0.0)
    assert metrics["grad_norm"] == pytest.approx(0.0)


def test_run_impala_optimizer_step_preserves_standard_amp_backoff_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=True,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert np.isnan(metrics["grad_norm"])


def test_run_impala_optimizer_step_preserves_auxiliary_amp_update_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(8.0)
    assert np.isnan(metrics["grad_norm"])


def test_begin_impala_update_scope_counts_normal_updates_and_checkpoint_metadata(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=1,
        profile_timers=True,
    )

    scope = begin_impala_update_scope(
        learner=learner,
        batch=_simple_training_batch(),
        count_learner_update=True,
        include_training_metrics=True,
        checkpoint_on_interval=True,
    )

    assert learner.update_count == 1
    assert learner.total_samples_processed == 2
    assert learner.policy_version == 1
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_1.json").is_file()
    assert scope.metrics["loss"] == pytest.approx(0.0)
    assert scope.metrics["entropy_coef"] == pytest.approx(float(learner.entropy_coef))
    assert scope.metrics["throughput_samples_per_sec"] >= 0.0
    assert scope.metrics["throughput_updates_per_sec"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics == {}


def test_finalize_impala_update_scope_merges_and_clears_profile_timers() -> None:
    learner = ImpalaLearner(profile_timers=True)
    cast(Any, learner)._active_timing_metrics = {"timer_custom_ms": 3.5}

    metrics = finalize_impala_update_scope(
        learner=learner,
        metrics={"loss": 1.0},
        started_at=time.perf_counter(),
    )

    assert metrics["loss"] == pytest.approx(1.0)
    assert metrics["timer_custom_ms"] == pytest.approx(3.5)
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_auxiliary_update_scope_preserves_update_count_and_training_metrics_policy() -> None:
    learner = ImpalaLearner(profile_timers=True)
    learner.update_count = 7

    scope = begin_impala_update_scope(
        learner=learner,
        batch=_simple_training_batch(),
        count_learner_update=False,
        include_training_metrics=False,
        checkpoint_on_interval=False,
    )

    assert learner.update_count == 7
    assert learner.total_samples_processed == 2
    assert scope.metrics == {}
    assert cast(Any, learner)._active_timing_metrics == {}


def test_set_impala_model_train_mode_sets_compiled_model_too() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    model.eval()
    compiled_model.eval()
    learner = ImpalaLearner(model=model, compiled_model=compiled_model)

    set_impala_model_train_mode(learner)

    assert model.training is True
    assert compiled_model.training is True


def test_build_scoped_impala_loss_sets_train_mode_times_and_preserves_outputs() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    learner = ImpalaLearner(model=model, compiled_model=compiled_model, profile_timers=True)
    model.eval()
    compiled_model.eval()
    timings: list[tuple[str, float]] = []
    cast(Any, learner)._record_timing_ms = lambda name, duration: timings.append((name, duration))
    loss = model.policy.weight.sum()
    metrics = {"custom_loss": 1.0}
    context = {"custom_context": torch.tensor(1.0)}
    calls: list[str] = []

    stage = build_scoped_impala_loss(
        learner=learner,
        loss_timer_name="learner_custom_loss",
        build_loss=lambda: (
            calls.append("loss") or loss,
            metrics,
            context,
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert compiled_model.training is True
    assert stage.loss is loss
    assert stage.metrics is metrics
    assert stage.context is context
    assert [name for name, _duration in timings] == ["learner_custom_loss"]
    assert timings[0][1] >= 0.0


def test_run_scoped_impala_optimizer_update_preserves_auxiliary_scope_and_timing_contract() -> None:
    model = nn.Linear(1, 1, bias=False)
    learner = ImpalaLearner(model=model, profile_timers=True)
    learner.update_count = 4
    model.eval()
    calls: list[str] = []

    metrics = run_scoped_impala_optimizer_update(
        learner=learner,
        batch=_simple_training_batch(),
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message="missing model",
            loss_timer_name="learner_custom_loss",
        ),
        build_loss=lambda: (
            calls.append("loss") or model.weight.sum(),
            {"custom_loss": 1.0},
            {"context": torch.tensor(1.0)},
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert learner.update_count == 4
    assert learner.total_samples_processed == 2
    assert metrics["custom_loss"] == pytest.approx(1.0)
    assert "grad_norm" in metrics
    assert metrics["timer_learner_custom_loss_ms"] >= 0.0
    assert metrics["timer_learner_backward_ms"] >= 0.0
    assert metrics["timer_learner_optimizer_ms"] >= 0.0
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_run_scoped_impala_optimizer_update_rejects_missing_model_before_loss_build() -> None:
    learner = ImpalaLearner(model=None)
    calls: list[str] = []

    with pytest.raises(ValueError, match="custom missing model"):
        run_scoped_impala_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
            spec=ScopedOptimizerUpdateSpec(
                missing_model_message="custom missing model",
                loss_timer_name="learner_custom_loss",
            ),
            build_loss=lambda: (
                calls.append("loss") or torch.tensor(1.0),
                {},
                {},
            ),
        )

    assert calls == []
    assert learner.update_count == 0
    assert learner.total_samples_processed == 0


def test_run_impala_auxiliary_optimizer_update_uses_auxiliary_loss_and_scoped_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"auxiliary": True}
    loss = torch.tensor(1.25)
    loss_metrics = {"auxiliary_metric": 1.25}
    loss_context = {"auxiliary_context": torch.tensor(2.0)}
    calls: list[tuple[str, Any]] = []

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert (
            kwargs["spec"].missing_model_message == "ImpalaLearner requires a model to run an auxiliary optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_auxiliary_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _auxiliary_loss_and_metrics=lambda source_batch: (
            calls.append(("auxiliary_loss", source_batch)) or loss,
            loss_metrics,
            loss_context,
        ),
    )
    monkeypatch.setattr(
        impala_auxiliary_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_auxiliary_update.run_impala_auxiliary_optimizer_update(learner=learner, batch=batch)

    assert result == {"loss": pytest.approx(1.25), "auxiliary_metric": 1.25}
    assert [name for name, _payload in calls] == ["scoped", "auxiliary_loss"]
    assert calls[1] == ("auxiliary_loss", batch)


def test_run_impala_auxiliary_optimizer_update_rejects_missing_model_before_auxiliary_loss() -> None:
    learner = SimpleNamespace(
        model=None,
        _auxiliary_loss_and_metrics=lambda _batch: pytest.fail("auxiliary loss should not be built"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an auxiliary optimizer step"):
        impala_auxiliary_update.run_impala_auxiliary_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
        )


def test_run_impala_normal_update_runs_training_step_diagnostics_logging_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"training": True}
    scope_metrics = {"loss": 0.0, "throughput": 10.0}
    vtrace_result = object()
    calls: list[tuple[str, Any]] = []

    def fake_begin_scope(**kwargs: Any) -> SimpleNamespace:
        calls.append(("begin", kwargs))
        assert kwargs == {
            "learner": learner,
            "batch": batch,
            "count_learner_update": True,
            "include_training_metrics": True,
            "checkpoint_on_interval": True,
        }
        return SimpleNamespace(started_at=12.5, metrics=scope_metrics)

    def fake_training_step(**kwargs: Any) -> dict[str, float]:
        calls.append(("training", kwargs))
        assert kwargs == {"learner": learner, "batch": batch}
        return {"loss": 1.5, "grad_norm": 0.25}

    def fake_summarize(**kwargs: Any) -> dict[str, float]:
        calls.append(("summarize", kwargs))
        assert kwargs == {"learner": learner, "batch": batch, "vtrace_result": vtrace_result}
        return {"vtrace_rho_p50": 0.75}

    def fake_log(**kwargs: Any) -> bool:
        calls.append(("log", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["metrics"] is scope_metrics
        assert kwargs["metrics"] == {
            "loss": 1.5,
            "throughput": 10.0,
            "grad_norm": 0.25,
            "vtrace_rho_p50": 0.75,
        }
        return True

    def fake_finalize(**kwargs: Any) -> dict[str, float]:
        calls.append(("finalize", kwargs))
        assert kwargs == {"learner": learner, "metrics": scope_metrics, "started_at": 12.5}
        return {"final_loss": scope_metrics["loss"], "final_vtrace": scope_metrics["vtrace_rho_p50"]}

    monkeypatch.setattr(impala_normal_update, "begin_impala_update_scope", fake_begin_scope)
    monkeypatch.setattr(impala_normal_update, "resolve_impala_update_vtrace_result", lambda source_batch: vtrace_result)
    monkeypatch.setattr(impala_normal_update, "has_impala_training_inputs", lambda source_batch: True)
    monkeypatch.setattr(impala_normal_update, "run_impala_training_optimizer_step", fake_training_step)
    monkeypatch.setattr(impala_normal_update, "summarize_precomputed_vtrace_update_metrics", fake_summarize)
    monkeypatch.setattr(impala_normal_update, "log_impala_update_metrics_if_due", fake_log)
    monkeypatch.setattr(impala_normal_update, "finalize_impala_update_scope", fake_finalize)

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"final_loss": 1.5, "final_vtrace": 0.75}
    assert [name for name, _payload in calls] == ["begin", "training", "summarize", "log", "finalize"]


def test_run_impala_normal_update_skips_training_step_without_training_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"metadata_only": True}
    scope_metrics = {"loss": 0.0}
    calls: list[str] = []

    monkeypatch.setattr(
        impala_normal_update,
        "begin_impala_update_scope",
        lambda **kwargs: calls.append("begin") or SimpleNamespace(started_at=3.0, metrics=scope_metrics),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "resolve_impala_update_vtrace_result",
        lambda source_batch: calls.append("vtrace") or None,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "has_impala_training_inputs",
        lambda source_batch: calls.append("has_training") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "run_impala_training_optimizer_step",
        lambda **_kwargs: pytest.fail("training optimizer step should be skipped"),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "summarize_precomputed_vtrace_update_metrics",
        lambda **kwargs: calls.append("summarize") or {"vtrace_rows": 0.0},
    )
    monkeypatch.setattr(
        impala_normal_update,
        "log_impala_update_metrics_if_due",
        lambda **kwargs: calls.append("log") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "finalize_impala_update_scope",
        lambda **kwargs: calls.append("finalize") or dict(kwargs["metrics"]),
    )

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"loss": 0.0, "vtrace_rows": 0.0}
    assert calls == ["begin", "vtrace", "has_training", "summarize", "log", "finalize"]


def test_run_impala_paired_swing_optimizer_update_validates_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=object())

    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        lambda **_kwargs: calls.append("scoped"),
    )

    with pytest.raises(ValueError, match="full_surface_top_action_retention_coef must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_top_action_retention_margin must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_margin=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_retention_batch is required"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=0.5,
        )

    assert calls == []


def test_run_impala_paired_swing_optimizer_update_composes_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"paired": True}
    retention_batch = {"retention": True}
    swing_loss = torch.tensor(2.0)
    retention_loss = torch.tensor(0.5)
    calls: list[tuple[str, Any]] = []

    def paired_swing_loss(source_batch: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("swing", (source_batch, kwargs)))
        return swing_loss, {"swing_metric": 2.0}, {"swing_context": "base"}

    def full_surface_retention(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("retention", (source_batch, kwargs)))
        return retention_loss, {"retention_metric": 0.5}, {"retention_context": "extra"}

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired-swing optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_swing_loss_and_metrics"
        loss, metrics, context = kwargs["build_loss"]()
        assert loss.item() == pytest.approx(2.5)
        assert metrics == {"swing_metric": 2.0, "retention_metric": 0.5}
        assert context == {"swing_context": "base", "retention_context": "extra"}
        return {"loss": float(loss.item()), **metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_swing_loss_and_metrics=paired_swing_loss,
        _paired_swing_full_surface_top_action_retention_loss_and_metrics=full_surface_retention,
    )
    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
        learner=learner,
        batch=batch,
        margin=1,
        coef=0.75,
        positive_action_source="teacher_positive",
        negative_action_source="learner_negative",
        loss_scope="span",
        compare_to="baseline",
        margin_retention_coef=0.25,
        margin_retention_margin=0.5,
        top_action_retention_coef=0.125,
        top_action_retention_margin=0.75,
        full_surface_retention_batch=retention_batch,
        full_surface_top_action_retention_coef=0.4,
        full_surface_top_action_retention_margin=0.6,
        full_surface_top_action_retention_mode="target_action",
    )

    assert result == {"loss": pytest.approx(2.5), "swing_metric": 2.0, "retention_metric": 0.5}
    assert [name for name, _payload in calls] == ["scoped", "swing", "retention"]
    assert calls[1][0] == "swing"
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "margin": 1.0,
        "coef": 0.75,
        "positive_action_source": "teacher_positive",
        "negative_action_source": "learner_negative",
        "loss_scope": "span",
        "compare_to": "baseline",
        "margin_retention_coef": 0.25,
        "margin_retention_margin": 0.5,
        "top_action_retention_coef": 0.125,
        "top_action_retention_margin": 0.75,
    }
    assert calls[2] == (
        "retention",
        (
            retention_batch,
            {
                "coef": 0.4,
                "margin": 0.6,
                "mode": "target_action",
            },
        ),
    )


def test_run_impala_paired_outcome_preference_optimizer_update_forwards_casted_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    loss = torch.tensor(1.75)
    loss_metrics = {"preference_metric": 1.75}
    loss_context = {"preference_context": torch.tensor(3.0)}
    calls: list[tuple[str, Any]] = []

    def paired_outcome_loss(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("preference_loss", (source_batch, kwargs)))
        return loss, loss_metrics, loss_context

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired outcome preference optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_outcome_preference_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=paired_outcome_loss,
    )
    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta="0.7",
        coef="0.25",
        aggregation=123,
        group_balance=1,
        retention_coef="0.5",
        retention_margin="0.125",
        retention_role=456,
        retention_reference_top_only=1,
        top_action_retention_coef="0.75",
        top_action_retention_margin="0.875",
        top_action_retention_role=789,
        top_action_retention_reference_top_only=1,
    )

    assert result == {"loss": pytest.approx(1.75), "preference_metric": 1.75}
    assert [name for name, _payload in calls] == ["scoped", "preference_loss"]
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "beta": 0.7,
        "coef": 0.25,
        "aggregation": "123",
        "group_balance": True,
        "retention_coef": 0.5,
        "retention_margin": 0.125,
        "retention_role": "456",
        "retention_reference_top_only": True,
        "top_action_retention_coef": 0.75,
        "top_action_retention_margin": 0.875,
        "top_action_retention_role": "789",
        "top_action_retention_reference_top_only": True,
    }


def test_run_impala_paired_outcome_preference_optimizer_update_uses_default_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    captured_kwargs: dict[str, Any] = {}
    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=lambda source_batch, **kwargs: (
            captured_kwargs.update(kwargs) or torch.tensor(0.5),
            {"preference_metric": 0.5},
            {},
        ),
    )

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        loss, metrics, _context = kwargs["build_loss"]()
        return {"loss": float(loss.item()), **metrics}

    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta=0.3,
        coef=0.2,
    )

    assert result == {"loss": pytest.approx(0.5), "preference_metric": 0.5}
    assert captured_kwargs == {
        "beta": 0.3,
        "coef": 0.2,
        "aggregation": "mean",
        "group_balance": False,
        "retention_coef": 0.0,
        "retention_margin": 0.0,
        "retention_role": "preferred",
        "retention_reference_top_only": False,
        "top_action_retention_coef": 0.0,
        "top_action_retention_margin": 0.0,
        "top_action_retention_role": "all",
        "top_action_retention_reference_top_only": False,
    }


def test_impala_update_training_input_helpers_preserve_missing_field_contract() -> None:
    learner = SimpleNamespace(
        _has_legal_actions=lambda batch: False,
        _has_raw_vtrace_inputs=lambda batch: False,
    )
    batch = {"obs": np.zeros((1, 1, 2), dtype=np.float32)}

    assert has_impala_training_inputs(batch) is True
    assert resolve_impala_update_vtrace_result(batch) is None
    assert missing_impala_training_input_fields(learner=learner, batch=batch) == [
        "actions",
        "legal_actions",
        "vtrace_result_or_raw_inputs",
    ]
    with pytest.raises(
        ValueError,
        match=(
            "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs "
            "for learner updates; missing actions, legal_actions, vtrace_result_or_raw_inputs"
        ),
    ):
        validate_impala_training_inputs(learner=learner, batch=batch)


def test_impala_update_training_input_helpers_accept_raw_vtrace_and_summarize_precomputed_targets() -> None:
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _has_legal_actions=lambda batch: True,
        _has_raw_vtrace_inputs=lambda batch: True,
    )
    vtrace_result = VTraceTargets(
        vs=np.zeros((2, 1), dtype=np.float32),
        pg_advantages=np.zeros((2, 1), dtype=np.float32),
        rhos=np.asarray([[0.5], [2.0]], dtype=np.float32),
    )
    batch = {
        "obs": np.zeros((2, 1, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.int64),
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": 1.5,
        "vtrace_c_bar": 0.75,
    }

    assert missing_impala_training_input_fields(learner=learner, batch=batch) == []
    validate_impala_training_inputs(learner=learner, batch=batch)
    assert (
        summarize_precomputed_vtrace_update_metrics(
            learner=learner,
            batch=batch,
            vtrace_result=None,
        )
        == {}
    )

    metrics = summarize_precomputed_vtrace_update_metrics(
        learner=learner,
        batch=batch,
        vtrace_result=vtrace_result,
    )

    assert metrics["vtrace_rho_p50"] == pytest.approx(1.25)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.5)


def test_log_impala_update_metrics_if_due_preserves_interval_and_timestamp_contract() -> None:
    calls: list[tuple[dict[str, float], dict[str, bool]]] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    learner = SimpleNamespace(
        logger=object(),
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda logged_metrics, logged_batch: calls.append((logged_metrics, logged_batch)),
    )

    logged = log_impala_update_metrics_if_due(
        learner=learner,
        batch=batch,
        metrics=metrics,
        now=123.5,
    )

    assert logged is True
    assert calls == [(metrics, batch)]
    assert learner.last_log_time == pytest.approx(123.5)
    assert learner.last_log_update == 6


def test_log_impala_update_metrics_if_due_skips_without_logger_or_interval() -> None:
    calls: list[str] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    no_logger = SimpleNamespace(
        logger=None,
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("no_logger"),
    )
    off_interval = SimpleNamespace(
        logger=object(),
        update_count=5,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("off_interval"),
    )

    assert log_impala_update_metrics_if_due(learner=no_logger, batch=batch, metrics=metrics, now=1.0) is False
    assert log_impala_update_metrics_if_due(learner=off_interval, batch=batch, metrics=metrics, now=1.0) is False
    assert calls == []
    assert no_logger.last_log_time == pytest.approx(0.0)
    assert no_logger.last_log_update == 0
    assert off_interval.last_log_time == pytest.approx(0.0)
    assert off_interval.last_log_update == 0


def test_run_impala_training_optimizer_step_validates_builds_loss_and_scales_nonfinite_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"training_step_batch": True}
    learner = SimpleNamespace(model=object())
    loss = torch.tensor(2.0)
    loss_metrics = {"loss": 2.0}
    loss_context = {"context": torch.tensor(1.0)}
    calls: list[str] = []

    def fake_validate_impala_training_inputs(*, learner: Any, batch: Any) -> None:
        assert learner is training_learner
        assert batch is training_batch
        calls.append("validate")

    def fake_build_scoped_impala_loss(*, learner: Any, loss_timer_name: str, build_loss: Any) -> SimpleNamespace:
        assert learner is training_learner
        assert loss_timer_name == "learner_loss_and_metrics"
        built_loss, built_metrics, built_context = build_loss()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        calls.append("build")
        return SimpleNamespace(loss=built_loss, metrics=built_metrics, context=built_context)

    def fake_run_impala_optimizer_step(**kwargs: Any) -> dict[str, float]:
        assert kwargs["learner"] is training_learner
        assert kwargs["batch"] is training_batch
        assert kwargs["loss"] is loss
        assert kwargs["base_metrics"] is loss_metrics
        assert kwargs["context"] is loss_context
        assert kwargs["scale_loss_on_nonfinite_gradients"] is True
        calls.append("optimizer")
        return {"loss": 2.0, "grad_norm": 0.5}

    training_learner = learner
    training_batch = batch
    learner._loss_and_metrics_with_context = lambda source_batch: (
        calls.append("loss") or loss,
        loss_metrics,
        loss_context,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        fake_validate_impala_training_inputs,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        fake_build_scoped_impala_loss,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "run_impala_optimizer_step",
        fake_run_impala_optimizer_step,
    )

    metrics = run_impala_training_optimizer_step(learner=learner, batch=batch)

    assert calls == ["validate", "loss", "build", "optimizer"]
    assert metrics == {"loss": 2.0, "grad_norm": 0.5}


def test_run_impala_training_optimizer_step_rejects_missing_model_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=None)

    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        lambda *, learner, batch: calls.append("validate"),
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        lambda **_kwargs: calls.append("build"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an optimizer step"):
        run_impala_training_optimizer_step(learner=learner, batch={})

    assert calls == ["validate"]


def _simple_training_batch() -> dict[str, object]:
    return {
        "obs": np.asarray(
            [
                [[1.0, 0.0]],
                [[0.5, -0.5]],
            ],
            dtype=np.float32,
        ),
        "actions": np.asarray(
            [
                [0],
                [1],
            ],
            dtype=np.int64,
        ),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }


def _packed_ids_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    offsets = [0]
    for row in np.asarray(mask, dtype=bool).reshape(-1, mask.shape[-1]):
        row_ids = np.flatnonzero(row).astype(np.uint32)
        ids.extend(int(value) for value in row_ids.tolist())
        offsets.append(len(ids))
    return np.asarray(ids, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32)


def _packed_meta_from_ids(action_catalog: ActionCatalog, packed_ids: np.ndarray) -> np.ndarray:
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    unused = np.iinfo(np.uint16).max
    rows = np.full((int(packed_ids.shape[0]), 4), unused, dtype=np.uint16)
    for row_index, action_id in enumerate(np.asarray(packed_ids, dtype=np.int64).tolist()):
        decoded = action_catalog.decode(int(action_id))
        rows[row_index, 0] = np.uint16(family_index[decoded.family])
        if decoded.hand_index is not None:
            rows[row_index, 1] = np.uint16(decoded.hand_index)
        if decoded.stage_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.stage_slot)
        if decoded.from_slot is not None:
            rows[row_index, 1] = np.uint16(decoded.from_slot)
        if decoded.to_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.to_slot)
        if decoded.slot is not None:
            rows[row_index, 1] = np.uint16(decoded.slot)
        if decoded.attack_type is not None:
            rows[row_index, 2] = np.uint16(attack_type_index[decoded.attack_type])
        if decoded.index is not None:
            rows[row_index, 1] = np.uint16(decoded.index)
    return rows


def _structured_metric_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 26,
                "pass_action_id": 25,
                "constants": [["MAX_HAND", 1], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 5},
                    {"name": "main_move", "base": 5, "count": 20},
                    {"name": "pass", "base": 25, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0]],
            }
        }
    )


def _teacher_aux_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 20,
                "pass_action_id": 19,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 3},
                    {"name": "main_move", "base": 13, "count": 6},
                    {"name": "pass", "base": 19, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def _teacher_aux_hand_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 22,
                "pass_action_id": 21,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "clock_from_hand", "base": 10, "count": 2},
                    {"name": "attack", "base": 12, "count": 3},
                    {"name": "main_move", "base": 15, "count": 6},
                    {"name": "pass", "base": 21, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def _mulligan_metric_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 9,
                "pass_action_id": 8,
                "constants": [["MAX_HAND", 4], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "mulligan_confirm", "base": 0, "count": 1},
                    {"name": "mulligan_select", "base": 1, "count": 4},
                    {"name": "attack", "base": 5, "count": 3},
                    {"name": "pass", "base": 8, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_impala_learner_writes_checkpoint_metadata_using_update_count(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
    )

    for _ in range(4):
        result = learner.update({})
        assert result["loss"] == 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_metadata_2.json").is_file()
    assert (checkpoint_dir / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_checkpoint_metadata_records_scope_update_and_policy_version(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=3,
    )

    for _ in range(3):
        learner.update({})

    checkpoint_metadata = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_metadata_3.json").read_text(encoding="utf-8")
    )
    assert checkpoint_metadata == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "policy_version": 1,
        "update_count": 3,
    }


def test_impala_learner_writes_fault_bundle_on_nonfinite_forward_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNLogitModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner forward_logits; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "forward_logits"
    assert payload["context"]["forward_logits_nonfinite_indices"]["data"] == [[0, 0, 0], [1, 0, 0]]


def test_impala_learner_writes_fault_bundle_on_nonfinite_gradients(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNGradientModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner gradients; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "gradients"
    assert "logit_bias" in payload["context"]["bad_gradient_names"]


def test_impala_learner_packed_legal_actions_match_dense_mask_loss() -> None:
    torch.manual_seed(0)
    dense_model = TinyPolicyValueModel()
    packed_model = TinyPolicyValueModel()
    packed_model.load_state_dict(dense_model.state_dict())
    dense_learner = ImpalaLearner(model=dense_model, pass_action_id=2)
    packed_learner = ImpalaLearner(model=packed_model, pass_action_id=2)

    legal_mask = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    actions = np.asarray([[0], [2]], dtype=np.int64)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": actions,
        "legal_mask": legal_mask,
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask)
    packed_batch = dict(batch)
    packed_batch["legal_actions"] = LegalActionBatch.from_packed(packed_ids, packed_offsets)
    packed_batch["legal_mask"] = None

    dense_loss, dense_metrics = dense_learner._loss_and_metrics(batch)
    packed_loss, packed_metrics = packed_learner._loss_and_metrics(packed_batch)

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_batch["legal_mask"] is None
    assert dense_metrics == pytest.approx(packed_metrics)


def test_resolve_impala_action_reductions_uses_packed_candidate_family_entropy() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5], [12]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19, 10, 11, 12, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3, 7], dtype=torch.long)
    packed_meta = torch.as_tensor(
        _packed_meta_from_ids(action_catalog, packed_ids.numpy().astype(np.uint32, copy=False)),
        dtype=torch.long,
    )
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0, -1.0, 0.5, 3.0, 0.0], dtype=torch.float32)
    timings: list[tuple[str, float]] = []

    reductions = resolve_impala_action_reductions(
        factorized_result=None,
        logits=None,
        packed_logits=packed_logits,
        legal_mask=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        actions=actions,
        entropy_scope="family",
        pass_action_id=action_catalog.pass_action_id,
        action_catalog=action_catalog,
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )
    expected_logp, _candidate_entropy = packed_scores_action_logp_and_entropy(
        packed_logits,
        packed_ids,
        packed_offsets,
        actions,
        pass_action_id=action_catalog.pass_action_id,
    )
    expected_family_entropy = packed_scores_family_entropy(
        packed_logits,
        packed_offsets,
        packed_meta,
        row_shape=actions.shape,
        family_count=len(action_catalog.families),
    )

    torch.testing.assert_close(reductions.action_logp, expected_logp)
    torch.testing.assert_close(reductions.entropy, expected_family_entropy)
    assert reductions.action_logp.shape == actions.shape
    assert reductions.entropy.shape == actions.shape
    assert torch.isfinite(reductions.action_logp).all()
    assert torch.isfinite(reductions.entropy).all()
    assert [name for name, _duration in timings] == ["learner_packed_reductions"]
    assert timings[0][1] >= 0.0


def test_resolve_impala_action_reductions_preserves_family_entropy_requirements() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="family entropy requires packed legal-action metadata and action_catalog"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=None,
            packed_logits=packed_logits,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )

    with pytest.raises(ValueError, match="family entropy requires packed candidate logits"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
            packed_logits=None,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )


def test_resolve_impala_action_reductions_preserves_factorized_requirement_error() -> None:
    with pytest.raises(ValueError, match="factorized learner path requires action_logp and entropy"):
        resolve_impala_action_reductions(
            factorized_result=SimpleNamespace(action_logp=torch.zeros((1, 1)), entropy=None),
            logits=None,
            packed_logits=None,
            legal_mask=None,
            packed_legal=None,
            actions=torch.zeros((1, 1), dtype=torch.long),
            entropy_scope="candidate",
            pass_action_id=None,
            action_catalog=None,
            record_timing_ms=lambda _name, _duration: None,
        )


class _TeacherTargetInputLearner:
    def __init__(self) -> None:
        self.teacher_public_heuristic_coef = 0.0
        self.teacher_public_nonpass_over_pass_coef = 0.0
        self.teacher_action_margin_coef = 0.0
        self.teacher_same_family_action_margin_coef = 0.0
        self.timings: list[tuple[str, float]] = []
        self.packed_public_target_calls = 0
        self.factorized_teacher_view_calls: list[bool] = []
        self.factorized_view = SimpleNamespace(row_has_candidates=torch.ones((1,), dtype=torch.bool))

    def _record_timing_ms(self, name: str, duration: float) -> None:
        self.timings.append((name, duration))

    def _packed_public_heuristic_target_logits(
        self,
        *,
        forward_model: Any,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        observation_context: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        del forward_model, obs, loss_mask, observation_context
        self.packed_public_target_calls += 1
        return torch.arange(int(packed_legal[0].numel()), dtype=torch.float32)

    def _factorized_public_heuristic_teacher_view(
        self,
        batch: Any,
        *,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        score_public_target: bool,
    ) -> tuple[Any, torch.Tensor | None]:
        del batch, obs, loss_mask, packed_legal
        self.factorized_teacher_view_calls.append(score_public_target)
        target_logits = torch.ones((3,), dtype=torch.float32) if score_public_target else None
        return self.factorized_view, target_logits


def test_prepare_impala_teacher_target_inputs_builds_packed_view_and_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)
    obs = torch.ones((1, 1, 2), dtype=torch.float32)
    packed_logits = torch.as_tensor([0.0, 1.0, -1.0], dtype=torch.float32)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=obs,
        logits=None,
        packed_logits=packed_logits,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context={"obs": obs.reshape(1, 2)},
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.as_tensor([0.0, 1.0, 2.0]))
    assert learner.packed_public_target_calls == 1
    assert [name for name, _duration in learner.timings] == ["learner_packed_view", "learner_public_heuristic_target"]


def test_prepare_impala_teacher_target_inputs_respects_teacher_aux_gate() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=torch.zeros((1, 1, _teacher_aux_catalog().action_space_size), dtype=torch.float32),
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=False,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is None
    assert learner.packed_public_target_calls == 0
    assert [name for name, _duration in learner.timings] == ["learner_packed_view"]


def test_resolve_impala_teacher_target_plan_names_candidate_target_gates() -> None:
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0

    disabled = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=False,
    )
    unsupported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )
    supported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )

    assert disabled.public_candidate_target_active is True
    assert disabled.factorized_candidate_teacher_view_active is False
    assert disabled.can_prepare_candidate_targets is False
    assert unsupported.can_prepare_candidate_targets is False
    assert supported.can_prepare_candidate_targets is True


def test_resolve_impala_teacher_target_plan_allows_factorized_margin_without_public_model_support() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )

    plan = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        teacher_aux_enabled=True,
    )

    assert plan.public_candidate_target_active is False
    assert plan.factorized_candidate_teacher_view_active is True
    assert plan.can_prepare_candidate_targets is True


def test_prepare_impala_teacher_target_inputs_scores_factorized_public_target_when_active() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.ones((3,), dtype=torch.float32))
    assert learner.factorized_teacher_view_calls == [True]


def test_prepare_impala_teacher_target_inputs_requests_factorized_margin_view_without_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is None
    assert learner.factorized_teacher_view_calls == [False]
    assert learner.timings == []


def test_compute_impala_objective_losses_uses_current_logp_for_retention_and_policy_logp_for_pg() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.zeros((2, 1), dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.25], [-2.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.ones((2, 1), dtype=torch.float32),
        values=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.zeros((2, 1), dtype=torch.float32),
        entropy=torch.zeros((2, 1), dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=None,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        trajectory_retention_valid=torch.as_tensor([[False], [True]], dtype=torch.bool),
        trajectory_retention_coef=0.5,
    )

    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.value_loss.item() == pytest.approx(0.0)
    assert result.trajectory_retention_metrics["trajectory_retention_loss"] == pytest.approx(2.0)
    assert result.trajectory_retention_metrics["trajectory_retention_weighted_loss"] == pytest.approx(1.0)
    assert result.total_loss.item() == pytest.approx(1.0)
    assert result.value_loss_mask.tolist() == [[1.0], [1.0]]


def test_compute_impala_objective_losses_respects_explicit_value_mask_and_entropy_term() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.as_tensor([[2.0], [10.0]], dtype=torch.float32),
        values=torch.as_tensor([[0.0], [3.0]], dtype=torch.float32),
        targets=torch.as_tensor([[2.0], [1.0]], dtype=torch.float32),
        entropy=torch.as_tensor([[0.25], [99.0]], dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=torch.as_tensor([[0.0], [1.0]], dtype=torch.float32),
        value_loss_coef=0.5,
        entropy_coef=0.1,
        trajectory_retention_valid=None,
        trajectory_retention_coef=0.0,
    )

    assert result.policy_loss.item() == pytest.approx(1.0)
    assert result.value_loss.item() == pytest.approx(4.0)
    assert result.entropy_mean.item() == pytest.approx(0.25)
    assert result.total_loss.item() == pytest.approx(2.975)
    assert result.value_loss_mask.tolist() == [[0.0], [1.0]]
    assert result.trajectory_retention_metrics == {}


def test_apply_impala_teacher_auxiliary_returns_unchanged_loss_when_inactive() -> None:
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    total_loss = torch.tensor(2.0)

    result = apply_impala_teacher_auxiliary(
        learner=object(),
        batch={},
        total_loss=total_loss,
        context=context,
        teacher_aux_active=False,
        logits=None,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=None,
        expected_shape=torch.Size((1, 1)),
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda _batch, _shape, _action_dim: pytest.fail(
            "inactive teacher aux must not resolve mask"
        ),
        batch_value=lambda batch, key: getattr(batch, key),
    )

    assert result.total_loss is total_loss
    assert result.teacher_metrics == {}
    assert list(context) == ["existing"]
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))


def test_apply_impala_teacher_auxiliary_resolves_dense_mask_for_packed_without_meta() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -1.0
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }
    context: dict[str, Any] = {}
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    result = apply_impala_teacher_auxiliary(
        learner=learner,
        batch=batch,
        total_loss=torch.tensor(1.0),
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=torch.Size((1, 1)),
        packed_legal=(packed_ids, packed_offsets, None),
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
            resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert resolver_calls == [(batch, torch.Size((1, 1)), action_catalog.action_space_size)]
    assert result.total_loss.item() > 1.0
    assert result.teacher_metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert "teacher_aux_loss" in context


def test_apply_impala_teacher_auxiliary_stage_maps_loss_inputs_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(2.0, dtype=torch.float32)
    resolved_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    learner = SimpleNamespace(_resolve_legal_mask=resolve_legal_mask)
    batch = {"teacher_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    logits = torch.zeros((2, 1, 7), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    public_targets = torch.zeros((2, 1, 7), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    inputs = SimpleNamespace(
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        values=values,
        packed_legal=packed_legal,
        teacher_aux_packed_view=packed_view,
        factorized_result=factorized_result,
        public_heuristic_target_logits=public_targets,
    )
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_apply_impala_teacher_auxiliary(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        assert kwargs["resolve_legal_mask"](batch, torch.Size((2, 1)), 7) is resolved_mask
        return SimpleNamespace(
            total_loss=kwargs["total_loss"] + torch.tensor(0.25, dtype=torch.float32),
            teacher_metrics={"teacher_valid_fraction": 0.5},
        )

    monkeypatch.setattr(
        impala_loss_teacher_stage,
        "apply_impala_teacher_auxiliary",
        fake_apply_impala_teacher_auxiliary,
    )

    result = apply_impala_teacher_auxiliary_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["total_loss"] is total_loss
    assert captured["context"] is context
    assert captured["teacher_aux_active"] is True
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] == "catalog"
    assert captured["expected_shape"] == values.shape
    assert captured["packed_legal"] is packed_legal
    assert captured["packed_view"] is packed_view
    assert captured["factorized_result"] is factorized_result
    assert captured["public_heuristic_target_logits"] is public_targets
    assert captured["batch_value"] is batch_value
    assert resolver_calls == [(batch, torch.Size((2, 1)), 7)]
    assert batch_values == []
    torch.testing.assert_close(result.total_loss, torch.tensor(2.25))
    assert result.teacher_metrics == {"teacher_valid_fraction": 0.5}


class _FiniteRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor]] = []

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        del batch, context
        self.calls.append((name, tensor))


def test_finalize_impala_loss_context_records_losses_and_finite_checks() -> None:
    learner = _FiniteRecorder()
    context: dict[str, Any] = {}
    factorized_result = SimpleNamespace(family_log_probs=torch.log_softmax(torch.ones((1, 1, 3)), dim=-1))
    policy_loss = torch.tensor(0.5)
    value_loss = torch.tensor(1.0)
    entropy_mean = torch.tensor(0.25)
    total_loss = torch.tensor(1.375)
    policy_anchor_loss = torch.tensor(0.125)

    finalize_impala_loss_context(
        learner=learner,
        batch={"batch": True},
        context=context,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        total_loss=total_loss,
        policy_anchor_loss=policy_anchor_loss,
        factorized_result=factorized_result,
    )

    assert context["policy_loss"] is not policy_loss
    torch.testing.assert_close(context["policy_loss"], policy_loss)
    torch.testing.assert_close(context["value_loss"], value_loss)
    torch.testing.assert_close(context["entropy_mean"], entropy_mean)
    torch.testing.assert_close(context["policy_anchor_loss"], policy_anchor_loss)
    torch.testing.assert_close(context["total_loss"], total_loss)
    torch.testing.assert_close(context["factorized_family_log_probs"], factorized_result.family_log_probs)
    assert [name for name, _tensor in learner.calls] == [
        "policy_loss",
        "value_loss",
        "entropy_mean",
        "total_loss",
    ]


def test_finalize_impala_loss_context_stage_maps_objective_anchor_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"context_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    factorized_result = SimpleNamespace(family_log_probs=torch.zeros((1, 1, 3), dtype=torch.float32))
    inputs = SimpleNamespace(
        context=context,
        factorized_result=factorized_result,
    )
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_anchor_loss = torch.tensor(0.25, dtype=torch.float32)
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_loss=policy_anchor_loss)
    captured: dict[str, Any] = {}

    def fake_finalize_impala_loss_context(**kwargs: Any) -> None:
        captured.update(kwargs)
        kwargs["context"]["finalized"] = torch.tensor(1.0)

    monkeypatch.setattr(
        impala_loss_context_stage,
        "finalize_impala_loss_context",
        fake_finalize_impala_loss_context,
    )

    result_context = finalize_impala_loss_context_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
    )

    assert result_context is context
    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["context"] is context
    assert captured["policy_loss"] is policy_loss
    assert captured["value_loss"] is value_loss
    assert captured["entropy_mean"] is entropy_mean
    assert captured["total_loss"] is total_loss
    assert captured["policy_anchor_loss"] is policy_anchor_loss
    assert captured["factorized_result"] is factorized_result
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))
    torch.testing.assert_close(context["finalized"], torch.tensor(1.0))


def test_assemble_impala_loss_core_metrics_maps_stage_outputs_to_metric_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    value_loss_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    actions = torch.tensor([[2], [3]], dtype=torch.long)
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-0.1], [-0.6]], dtype=torch.float32)
    rewards = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    advantages = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.75], [-0.25]], dtype=torch.float32)
    rhos = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 5), dtype=torch.bool)
    packed_legal = (
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    batch = {"metric_stage_batch": True}
    resolved_mask = torch.ones_like(legal_mask)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    timing_calls: list[tuple[str, float]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    def record_timing(name: str, duration: float) -> None:
        timing_calls.append((name, duration))

    learner = SimpleNamespace(
        entropy_scope="family",
        pass_action_id=4,
        _resolve_legal_mask=resolve_legal_mask,
        _record_timing_ms=record_timing,
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((2, 1, 3), dtype=torch.float32),
        loss_mask=loss_mask,
        actions=actions,
        emit_structured_metrics=True,
        logits=logits,
        legal_mask=legal_mask,
        packed_legal=packed_legal,
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        value_loss_mask=value_loss_mask,
        trajectory_retention_metrics={"trajectory_retention_rows": 2.0},
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25})
    teacher_finalization = SimpleNamespace(teacher_metrics={"teacher_aux_loss": 0.5})
    resolved_vtrace = SimpleNamespace(
        behavior_logp_for_mask=behavior_logp,
        rewards_for_metrics=rewards,
        advantages=advantages,
        targets=targets,
        rhos_for_metrics=rhos,
    )
    clip_config = SimpleNamespace(rho_bar=1.5, c_bar=1.25)
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_assemble_impala_loss_metrics(
        request: ImpalaMetricAssemblyRequest,
        *,
        batch_value: Any,
        record_timing_ms: Any,
    ) -> dict[str, float]:
        captured["request"] = request
        captured["batch_value"] = batch_value
        captured["record_timing_ms"] = record_timing_ms
        assert request.resolve_legal_mask is not None
        assert request.resolve_legal_mask(batch, torch.Size((2, 1)), 5) is resolved_mask
        return {"loss": 3.0, "metric_stage": 1.0}

    monkeypatch.setattr(
        impala_loss_metrics_stage,
        "assemble_impala_loss_metrics",
        fake_assemble_impala_loss_metrics,
    )

    metrics = assemble_impala_loss_core_metrics(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
        teacher_finalization=cast(Any, teacher_finalization),
        resolved_vtrace=resolved_vtrace,
        clip_config=clip_config,
        action_logp=action_logp,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    request = cast(ImpalaMetricAssemblyRequest, captured["request"])
    assert metrics == {"loss": 3.0, "metric_stage": 1.0}
    assert captured["batch_value"] is batch_value
    assert captured["record_timing_ms"] is record_timing
    assert request.total_loss is total_loss
    assert request.policy_loss is policy_loss
    assert request.value_loss is value_loss
    assert request.entropy_mean is entropy_mean
    assert request.entropy_scope == "family"
    assert request.loss_mask is loss_mask
    assert request.value_loss_mask is value_loss_mask
    assert request.actions is actions
    assert request.action_logp is action_logp
    assert request.behavior_logp_for_mask is behavior_logp
    assert request.rewards_for_metrics is rewards
    assert request.advantages is advantages
    assert request.targets is targets
    assert request.rhos_for_metrics is rhos
    assert request.rho_bar == pytest.approx(1.5)
    assert request.c_bar == pytest.approx(1.25)
    assert request.action_catalog == "catalog"
    assert request.pass_action_id == 4
    assert request.trajectory_retention_metrics == {"trajectory_retention_rows": 2.0}
    assert request.policy_anchor_metrics == {"policy_anchor_weighted_loss": 0.25}
    assert request.teacher_metrics == {"teacher_aux_loss": 0.5}
    assert request.emit_structured_metrics is True
    assert request.logits is logits
    assert request.legal_mask is legal_mask
    assert request.packed_legal is packed_legal
    assert request.packed_view is packed_view
    assert request.factorized_result is factorized_result
    assert request.batch is batch
    assert request.expected_shape == torch.Size((2, 1))
    assert request.action_dim == 5
    assert resolver_calls == [(batch, torch.Size((2, 1)), 5)]
    assert batch_values == []
    assert timing_calls == []


def test_assemble_impala_loss_metrics_preserves_base_metric_inputs_and_backfill_fields() -> None:
    batch = {
        "terminal_outcome_backfill_count": 3,
        "terminal_outcome_backfill_total_micros": 12.5,
        "terminal_outcome_trace_backfill_count": 2,
        "terminal_outcome_trace_backfill_total_micros": 7.25,
    }

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(1.5),
            policy_loss=torch.tensor(0.25),
            value_loss=torch.tensor(2.0),
            entropy_mean=torch.tensor(0.125),
            entropy_scope="candidate",
            loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
            value_loss_mask=torch.as_tensor([[1.0], [1.0]], dtype=torch.float32),
            actions=torch.as_tensor([[0], [1]], dtype=torch.long),
            action_logp=torch.as_tensor([[-0.2], [-0.3]], dtype=torch.float32),
            behavior_logp_for_mask=torch.as_tensor([[-0.1], [-0.5]], dtype=torch.float32),
            rewards_for_metrics=torch.as_tensor([[1.0], [-1.0]], dtype=torch.float32),
            advantages=torch.as_tensor([[0.5], [-0.25]], dtype=torch.float32),
            targets=torch.as_tensor([[1.25], [-0.75]], dtype=torch.float32),
            rhos_for_metrics=torch.as_tensor([[1.0], [2.0]], dtype=torch.float32),
            rho_bar=1.5,
            c_bar=1.25,
            action_catalog=object(),
            pass_action_id=1,
            trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
            policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
            teacher_metrics={"teacher_aux_loss": 0.5},
            emit_structured_metrics=True,
            batch=batch,
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda _name, _duration: pytest.fail("non-structured catalog must not summarize"),
    )

    assert metrics["loss"] == pytest.approx(1.5)
    assert metrics["policy_loss"] == pytest.approx(0.25)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert metrics["entropy"] == pytest.approx(0.125)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(3.0)
    assert metrics["terminal_outcome_backfill_total_micros"] == pytest.approx(12.5)
    assert metrics["terminal_outcome_trace_backfill_count"] == pytest.approx(2.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(7.25)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_aux_loss"] == pytest.approx(0.5)
    assert "structured_exact_action_concentration" not in metrics


def test_assemble_impala_loss_metrics_merges_structured_summary_with_dense_fallback() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 7, action_catalog.pass_action_id]] = True
    legal_mask[1, 0, [4, 7, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, action_catalog.pass_action_id] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, action_catalog.pass_action_id] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = {"marker": True}

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(0.0),
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy_mean=torch.tensor(0.0),
            entropy_scope="family",
            loss_mask=torch.ones((2, 1), dtype=torch.float32),
            value_loss_mask=torch.ones((2, 1), dtype=torch.float32),
            actions=torch.as_tensor([[0], [4]], dtype=torch.long),
            action_logp=torch.zeros((2, 1), dtype=torch.float32),
            behavior_logp_for_mask=None,
            rewards_for_metrics=torch.zeros((2, 1), dtype=torch.float32),
            advantages=torch.zeros((2, 1), dtype=torch.float32),
            targets=torch.zeros((2, 1), dtype=torch.float32),
            rhos_for_metrics=torch.ones((2, 1), dtype=torch.float32),
            rho_bar=1.0,
            c_bar=1.0,
            action_catalog=action_catalog,
            pass_action_id=action_catalog.pass_action_id,
            emit_structured_metrics=True,
            logits=logits,
            legal_mask=None,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=action_catalog.action_space_size,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    expected_structured = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    assert resolver_calls == [(batch, torch.Size((2, 1)), action_catalog.action_space_size)]
    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["structured_exact_action_concentration"] == pytest.approx(
        expected_structured["structured_exact_action_concentration"]
    )
    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(
        expected_structured["structured_main_move_0_2_top1_rate"]
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]


def test_build_impala_forward_context_detaches_outputs_and_checks_finiteness() -> None:
    calls: list[tuple[str, torch.Tensor, Any, dict[str, Any]]] = []
    learner = SimpleNamespace(
        _ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append((name, tensor, batch, context))
    )
    batch = {"forward_batch": True}
    logits = torch.ones((2, 1, 3), dtype=torch.float32, requires_grad=True)
    packed_logits = torch.arange(4, dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        values=values,
    )

    context = build_impala_forward_context(
        learner=learner,
        batch=batch,
        forward_result=forward_result,
    )

    torch.testing.assert_close(context["logits"], logits)
    torch.testing.assert_close(context["packed_logits"], packed_logits)
    torch.testing.assert_close(context["values"], values)
    assert context["logits"].requires_grad is False
    assert context["packed_logits"].requires_grad is False
    assert context["values"].requires_grad is False
    assert [
        (name, tensor, source_batch, call_context is context) for name, tensor, source_batch, call_context in calls
    ] == [
        ("forward_logits", logits, batch, True),
        ("forward_packed_logits", packed_logits, batch, True),
        ("forward_values", values, batch, True),
    ]


def test_build_impala_forward_context_skips_absent_logits_but_checks_values() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append(name))
    values = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)

    context = build_impala_forward_context(
        learner=learner,
        batch={},
        forward_result=SimpleNamespace(logits=None, packed_logits=None, values=values),
    )

    assert context["logits"] is None
    assert context["packed_logits"] is None
    torch.testing.assert_close(context["values"], values)
    assert context["values"].requires_grad is False
    assert calls == ["forward_values"]


def test_prepare_impala_loss_teacher_target_inputs_maps_forward_state_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"teacher_target_batch": True}
    forward_model = object()
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_logits = torch.arange(4, dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_observation_context = {"encoded": torch.ones((2, 1), dtype=torch.float32)}
    masks = SimpleNamespace(loss_mask=loss_mask)
    forward_flags = SimpleNamespace(emit_structured_metrics=False, teacher_aux_active=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_observation_context=forward_observation_context,
    )
    packed_view = object()
    teacher_view = object()
    public_targets = torch.ones((4,), dtype=torch.float32)
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_view,
            public_heuristic_target_logits=public_targets,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    result = prepare_impala_loss_teacher_target_inputs(
        learner=learner,
        batch=batch,
        forward_model=forward_model,
        obs=obs,
        masks=masks,
        forward_flags=forward_flags,
        forward_result=forward_result,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["forward_model"] is forward_model
    assert captured["obs"] is obs
    assert captured["logits"] is logits
    assert captured["packed_logits"] is packed_logits
    assert captured["packed_legal"] is packed_legal
    assert captured["loss_mask"] is loss_mask
    assert captured["factorized_result"] is factorized_result
    assert captured["forward_observation_context"] is forward_observation_context
    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is True
    assert result.packed_view is packed_view
    assert result.teacher_aux_packed_view is teacher_view
    assert result.public_heuristic_target_logits is public_targets


def test_prepare_impala_loss_teacher_target_inputs_needs_packed_view_for_structured_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=None,
            teacher_aux_packed_view=None,
            public_heuristic_target_logits=None,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    prepare_impala_loss_teacher_target_inputs(
        learner=object(),
        batch={},
        forward_model=object(),
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        masks=SimpleNamespace(loss_mask=torch.ones((1, 1), dtype=torch.float32)),
        forward_flags=SimpleNamespace(emit_structured_metrics=True, teacher_aux_active=False),
        forward_result=SimpleNamespace(
            logits=None,
            packed_logits=None,
            packed_legal=None,
            factorized_result=None,
            forward_observation_context=None,
        ),
    )

    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is False


def test_resolve_impala_loss_batch_inputs_prefers_compiled_forward_model_and_expected_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    model = object()
    compiled_model = object()
    batch = {
        "vtrace_result": "vtrace",
        "obs": "raw_obs",
        "actions": "raw_actions",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        calls.append(("batch_value", key))
        return source_batch.get(key)

    learner = SimpleNamespace(
        model=model,
        compiled_model=compiled_model,
        _require_obs=lambda value: calls.append(("require_obs", value)) or obs,
        _require_actions=lambda value, *, expected_shape: (
            calls.append(("require_actions", (value, expected_shape))) or actions
        ),
        _resolve_packed_legal_actions_with_meta=lambda source_batch, *, expected_shape: (
            calls.append(("packed_legal", (source_batch, expected_shape))) or packed_legal
        ),
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
    )

    assert result.vtrace_result == "vtrace"
    assert result.obs is obs
    assert result.actions is actions
    assert result.packed_legal is packed_legal
    assert result.forward_model is compiled_model
    assert calls == [
        ("batch_value", "vtrace_result"),
        ("batch_value", "obs"),
        ("require_obs", "raw_obs"),
        ("batch_value", "actions"),
        ("require_actions", ("raw_actions", torch.Size((2, 1)))),
        ("packed_legal", (batch, torch.Size((2, 1)))),
    ]


def test_resolve_impala_loss_batch_inputs_falls_back_to_base_model() -> None:
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    actions = torch.zeros((1, 1), dtype=torch.long)
    model = object()
    learner = SimpleNamespace(
        model=model,
        compiled_model=None,
        _require_obs=lambda _value: obs,
        _require_actions=lambda _value, *, expected_shape: actions,
        _resolve_packed_legal_actions_with_meta=lambda _batch, *, expected_shape: None,
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch={"obs": object(), "actions": object()},
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert result.forward_model is model
    assert result.packed_legal is None


def test_assemble_impala_loss_inputs_preserves_stage_outputs_by_identity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    reset_before_step = torch.zeros((2, 1), dtype=torch.bool)
    trajectory_retention_valid = torch.ones((2, 1), dtype=torch.float32)
    logits = torch.zeros((2, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((3,), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    public_target_logits = torch.full((3,), 0.25, dtype=torch.float32)
    original_packed_legal = (torch.as_tensor([0]), torch.as_tensor([0, 1]), None)
    forward_packed_legal = (torch.as_tensor([1, 2, 3]), torch.as_tensor([0, 3]), None)
    forward_model = object()
    factorized_result = object()
    observation_context = {"obs": obs.reshape(-1, obs.shape[-1])}
    context = {"logits": logits, "values": values}
    packed_view = cast(Any, object())
    teacher_aux_packed_view = cast(Any, object())

    assembled = assemble_impala_loss_inputs(
        batch_inputs=ImpalaLossBatchInputs(
            vtrace_result="vtrace",
            obs=obs,
            actions=actions,
            packed_legal=original_packed_legal,
            forward_model=forward_model,
        ),
        masks=ImpalaLossMasks(
            loss_mask=loss_mask,
            reset_before_step=reset_before_step,
            trajectory_retention_valid=trajectory_retention_valid,
            trajectory_retention_active=None,
        ),
        forward_flags=ImpalaLossForwardFlags(
            teacher_aux_active=True,
            emit_structured_metrics=True,
            restrict_packed_policy_rows=False,
        ),
        forward_result=ImpalaPolicyForwardResult(
            factorized_result=factorized_result,
            packed_legal=forward_packed_legal,
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            forward_observation_context=observation_context,
        ),
        legal_mask=legal_mask,
        teacher_target_inputs=ImpalaTeacherTargetInputs(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_aux_packed_view,
            public_heuristic_target_logits=public_target_logits,
        ),
        context=context,
    )

    assert assembled.vtrace_result == "vtrace"
    assert assembled.obs is obs
    assert assembled.actions is actions
    assert assembled.packed_legal is forward_packed_legal
    assert assembled.packed_legal is not original_packed_legal
    assert assembled.forward_model is forward_model
    assert assembled.loss_mask is loss_mask
    assert assembled.reset_before_step is reset_before_step
    assert assembled.trajectory_retention_valid is trajectory_retention_valid
    assert assembled.teacher_aux_active is True
    assert assembled.emit_structured_metrics is True
    assert assembled.factorized_result is factorized_result
    assert assembled.logits is logits
    assert assembled.packed_logits is packed_logits
    assert assembled.values is values
    assert assembled.forward_observation_context is observation_context
    assert assembled.legal_mask is legal_mask
    assert assembled.packed_view is packed_view
    assert assembled.teacher_aux_packed_view is teacher_aux_packed_view
    assert assembled.public_heuristic_target_logits is public_target_logits
    assert assembled.context is context


def test_prepare_impala_loss_inputs_restricts_packed_forward_to_policy_and_retention_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.4,
    )
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }
    captured_policy_masks: list[torch.Tensor | None] = []

    def fake_forward(
        obs: torch.Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: Any = None,
        policy_train_mask: torch.Tensor | None = None,
        reset_before_step: torch.Tensor | None = None,
        opponent_context_index: Any = None,
    ) -> SimpleNamespace:
        del initial_hidden_state, to_play_seat, actor, legal_actions, reset_before_step, opponent_context_index
        captured_policy_masks.append(None if policy_train_mask is None else policy_train_mask.detach().clone())
        return SimpleNamespace(
            logits=None,
            packed_logits=torch.zeros((int(packed_ids.shape[0]),), dtype=torch.float32),
            values=torch.zeros(obs.shape[:2], dtype=torch.float32),
            observation_context={"rows": obs.reshape(-1, obs.shape[-1])},
        )

    cast(Any, learner)._forward_time_major = fake_forward

    prepared = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))

    assert prepared.packed_legal is not None
    assert prepared.legal_mask is None
    assert prepared.teacher_aux_active is False
    assert prepared.emit_structured_metrics is False
    assert captured_policy_masks
    assert captured_policy_masks[0] is not None
    assert captured_policy_masks[0].tolist() == [[1.0], [1.0]]
    assert prepared.context["packed_logits"].shape == (int(packed_ids.shape[0]),)
    assert prepared.context["values"].tolist() == [[0.0], [0.0]]


def test_resolve_impala_loss_masks_converts_reset_and_retention_activity() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), trajectory_retention_coef=0.4)
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    batch = {
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "reset_before_step": np.asarray([[False], [True]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }

    masks = resolve_impala_loss_masks(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.tolist() == [[1.0], [0.0]]
    assert masks.reset_before_step is not None
    assert masks.reset_before_step.dtype == torch.bool
    assert masks.reset_before_step.tolist() == [[False], [True]]
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[0.0], [1.0]]
    assert masks.trajectory_retention_active is not None
    assert masks.trajectory_retention_active.dtype == torch.bool
    assert masks.trajectory_retention_active.tolist() == [[False], [True]]


def test_resolve_impala_loss_masks_defaults_policy_mask_and_disables_retention_activity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float64)
    batch = {
        "trajectory_retention_valid": np.asarray([[True], [False]], dtype=np.bool_),
    }
    calls: list[tuple[str, Any]] = []
    learner = SimpleNamespace(
        trajectory_retention_coef=0.0,
        _optional_time_major_loss_mask=lambda value, *, expected_shape, like: (
            calls.append(("mask", (value, expected_shape, like.shape, like.dtype)))
            or (torch.as_tensor(value, dtype=torch.float32) if value is not None else None)
        ),
    )

    masks = resolve_impala_loss_masks_stage(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.dtype == obs.dtype
    assert masks.loss_mask.device == obs.device
    assert masks.loss_mask.tolist() == [[1.0], [1.0]]
    assert masks.reset_before_step is None
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[1.0], [0.0]]
    assert masks.trajectory_retention_active is None
    assert calls[0] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[1] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[2][0] == "mask"
    assert calls[2][1][0] is batch["trajectory_retention_valid"]
    assert calls[2][1][1:] == (torch.Size((2, 1)), torch.Size((2, 1)), torch.float64)


def test_resolve_impala_loss_forward_flags_only_restricts_safe_packed_policy_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_legal = (
        torch.as_tensor([0, action_catalog.pass_action_id], dtype=torch.long),
        torch.as_tensor([0, 2], dtype=torch.long),
        torch.as_tensor(_packed_meta_from_ids(action_catalog, np.asarray([0, action_catalog.pass_action_id]))),
    )
    loss_mask = torch.as_tensor([[1.0], [0.0]], dtype=torch.float32)

    plain = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
    )
    teacher_model = TinyPolicyValueModel(action_dim=action_catalog.action_space_size)
    teacher_model.action_catalog = action_catalog
    teacher = ImpalaLearner(model=teacher_model, teacher_action_coef=0.5, structured_metrics_mode="off")
    structured = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="full",
    )

    plain_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=packed_legal, loss_mask=loss_mask)
    teacher_flags = resolve_impala_loss_forward_flags(learner=teacher, packed_legal=packed_legal, loss_mask=loss_mask)
    structured_flags = resolve_impala_loss_forward_flags(
        learner=structured,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
    )
    dense_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=None, loss_mask=loss_mask)

    assert plain_flags.teacher_aux_active is False
    assert plain_flags.emit_structured_metrics is False
    assert plain_flags.restrict_packed_policy_rows is True
    assert teacher_flags.teacher_aux_active is True
    assert teacher_flags.restrict_packed_policy_rows is False
    assert structured_flags.emit_structured_metrics is True
    assert structured_flags.restrict_packed_policy_rows is False
    assert dense_flags.restrict_packed_policy_rows is False


def test_evaluate_impala_policy_forward_uses_factorized_path_without_dense_forward() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1]], dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True]], dtype=torch.bool)
    original_packed = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)
    resolved_packed = (torch.as_tensor([1]), torch.as_tensor([0, 1]), None)
    factorized_result = SimpleNamespace(values=torch.as_tensor([[0.25], [0.5]], dtype=torch.float32))
    calls: list[tuple[str, Any]] = []

    def should_use_factorized(forward_model: object, *, packed_legal: object) -> bool:
        calls.append(("should_use_factorized", (forward_model, packed_legal)))
        return True

    def evaluate_factorized(
        source_batch: object,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        extra_active_mask: torch.Tensor | None,
    ) -> tuple[SimpleNamespace, tuple[torch.Tensor, torch.Tensor, None]]:
        calls.append(("evaluate_factorized", (source_batch, obs, actions, extra_active_mask)))
        return factorized_result, resolved_packed

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=should_use_factorized,
        _evaluate_factorized_time_major=evaluate_factorized,
    )
    forward_model = object()
    batch = object()

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=lambda _source, key: pytest.fail(f"unexpected batch_value({key})"),
        forward_model=forward_model,
        obs=obs,
        actions=actions,
        packed_legal=original_packed,
        loss_mask=loss_mask,
        reset_before_step=None,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    assert result.factorized_result is factorized_result
    assert result.packed_legal is resolved_packed
    assert result.logits is None
    assert result.packed_logits is None
    assert result.values is factorized_result.values
    assert result.forward_observation_context is None
    assert calls == [
        ("should_use_factorized", (forward_model, original_packed)),
        ("evaluate_factorized", (batch, obs, actions, retention_active)),
    ]


def test_evaluate_impala_policy_forward_forwards_dense_kwargs_and_restricts_rows() -> None:
    obs = torch.zeros((3, 1, 2), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1], [0]], dtype=torch.long)
    loss_mask = torch.as_tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    reset_before_step = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    logits = torch.zeros((3, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((5,), dtype=torch.float32)
    values = torch.as_tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    observation_context = {"rows": obs.reshape(-1, obs.shape[-1])}
    batch = {
        "initial_hidden_state": "hidden",
        "to_play_seat": "seat",
        "actor": "actor",
        "legal_actions": "legal",
        "opponent_context_index": "opponent",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Mapping[str, object], key: str) -> object:
        calls.append(("batch_value", key))
        return source_batch[key]

    def forward_time_major(
        forward_obs: torch.Tensor,
        *,
        initial_hidden_state: object,
        to_play_seat: object,
        actor: object,
        legal_actions: object,
        policy_train_mask: torch.Tensor | None,
        reset_before_step: torch.Tensor | None,
        opponent_context_index: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "forward",
                (
                    forward_obs,
                    initial_hidden_state,
                    to_play_seat,
                    actor,
                    legal_actions,
                    policy_train_mask,
                    reset_before_step,
                    opponent_context_index,
                ),
            )
        )
        return SimpleNamespace(
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            observation_context=observation_context,
        )

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=lambda _forward_model, *, packed_legal: False,
        _forward_time_major=forward_time_major,
    )
    packed_legal = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
        forward_model=object(),
        obs=obs,
        actions=actions,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
        reset_before_step=reset_before_step,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    forward_call = calls[-1]
    assert forward_call[0] == "forward"
    forwarded_mask = forward_call[1][5]
    assert isinstance(forwarded_mask, torch.Tensor)
    assert forwarded_mask.dtype == loss_mask.dtype
    assert forwarded_mask.tolist() == [[1.0], [1.0], [0.0]]
    assert forward_call[1][0] is obs
    assert forward_call[1][1:5] == ("hidden", "seat", "actor", "legal")
    assert forward_call[1][6] is reset_before_step
    assert forward_call[1][7] == "opponent"
    assert calls[:5] == [
        ("batch_value", "initial_hidden_state"),
        ("batch_value", "to_play_seat"),
        ("batch_value", "actor"),
        ("batch_value", "legal_actions"),
        ("batch_value", "opponent_context_index"),
    ]
    assert result.factorized_result is None
    assert result.packed_legal is packed_legal
    assert result.logits is logits
    assert result.packed_logits is packed_logits
    assert result.values is values
    assert result.forward_observation_context is observation_context


def test_resolve_impala_dense_legal_mask_returns_none_for_packed_legal_without_resolving() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: calls.append("resolve"))
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        None,
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch={},
        obs=obs,
        packed_legal=packed_legal,
        logits=logits,
    )

    assert result is None
    assert calls == []


def test_resolve_impala_dense_legal_mask_resolves_and_validates_dense_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    batch = {"dense": True}
    calls: list[tuple[Any, torch.Size, int]] = []
    learner = SimpleNamespace(
        _resolve_legal_mask=lambda source_batch, *, expected_shape, action_dim: (
            calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        )
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch=batch,
        obs=obs,
        packed_legal=None,
        logits=logits,
    )

    assert result is legal_mask
    assert calls == [(batch, torch.Size((2, 1)), 5)]


def test_resolve_impala_dense_legal_mask_rejects_missing_logits_and_shape_mismatch() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: torch.ones((2, 1, 4), dtype=torch.bool))

    with pytest.raises(ValueError, match="dense learner path requires dense logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=None,
        )
    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=logits,
        )


def test_prepare_impala_loss_inputs_rejects_dense_legal_mask_shape_mismatch() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
    }

    def bad_legal_mask(_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        del expected_shape, action_dim
        return torch.ones((1, 1, 2), dtype=torch.bool)

    cast(Any, learner)._resolve_legal_mask = bad_legal_mask

    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))


def test_resolve_impala_loss_action_reductions_attaches_detached_context_and_checks_finiteness() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()
    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    finite_calls: list[str] = []

    def record_finite(name: str, tensor: torch.Tensor, *, batch: Any, context: dict[str, Any]) -> None:
        del tensor, batch, context
        finite_calls.append(name)

    cast(Any, learner)._ensure_finite_tensor = record_finite

    reductions = resolve_impala_loss_action_reductions(
        learner=learner,
        batch=batch,
        loss_inputs=inputs,
    )

    assert reductions.context is inputs.context
    assert reductions.context["action_logp"].shape == torch.Size((2, 1))
    assert reductions.context["entropy"].shape == torch.Size((2, 1))
    assert reductions.context["action_logp"].requires_grad is False
    assert reductions.context["entropy"].requires_grad is False
    assert finite_calls == ["action_logp", "entropy"]


def test_resolve_impala_vtrace_clip_config_prefers_batch_overrides_then_learner_defaults() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=1.5, vtrace_c_bar=0.75)

    defaults = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={},
        batch_value=lambda source, key: source.get(key),
    )
    overrides = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={"vtrace_rho_bar": 2.25, "vtrace_c_bar": 0.5},
        batch_value=lambda source, key: source.get(key),
    )

    assert defaults.rho_bar == pytest.approx(1.5)
    assert defaults.c_bar == pytest.approx(0.75)
    assert overrides.rho_bar == pytest.approx(2.25)
    assert overrides.c_bar == pytest.approx(0.5)


def test_attach_resolved_vtrace_context_and_value_mask_keep_detached_loss_diagnostics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    values = torch.zeros((2, 1), dtype=torch.float32)
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    resolved_vtrace = SimpleNamespace(
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    context: dict[str, Any] = {}

    attach_resolved_vtrace_context(
        context=context,
        resolved_vtrace=resolved_vtrace,
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True),
    )
    value_mask = resolve_impala_value_loss_mask(
        learner=learner,
        batch=batch,
        expected_shape=torch.Size((2, 1)),
        like=values,
        batch_value=lambda source, key: source.get(key),
    )

    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False
    assert value_mask is not None
    assert value_mask.tolist() == [[0.0], [1.0]]


def test_compute_impala_vtrace_stage_resolves_targets_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32, requires_grad=True)
    resolved_action_logp = torch.tensor([[-0.3], [-0.8]], dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        vtrace_result="vtrace-result",
        values=values,
        loss_mask=loss_mask,
        context=context,
    )
    batch = {"vtrace_rho_bar": 2.0, "vtrace_c_bar": 0.5}
    float_target = object()
    resolve_bootstrap_value = object()
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _float_target=float_target,
        _resolve_vtrace_bootstrap_value=resolve_bootstrap_value,
    )
    resolved_vtrace = SimpleNamespace(
        action_logp=resolved_action_logp,
        behavior_logp_for_mask=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    captured: dict[str, Any] = {}

    def fake_resolve_impala_vtrace_targets(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return resolved_vtrace

    monkeypatch.setattr(
        impala_loss_vtrace_stage,
        "resolve_impala_vtrace_targets",
        fake_resolve_impala_vtrace_targets,
    )
    batch_value_calls: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_value_calls.append((source_batch, key))
        return source_batch.get(key)

    stage = compute_impala_vtrace_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        action_logp=action_logp,
        batch_value=batch_value,
    )

    assert stage.retention_action_logp is action_logp
    assert stage.action_logp is resolved_action_logp
    assert stage.clip_config.rho_bar == pytest.approx(2.0)
    assert stage.clip_config.c_bar == pytest.approx(0.5)
    assert stage.resolved_vtrace is resolved_vtrace
    assert captured["batch"] is batch
    assert captured["vtrace_result"] == "vtrace-result"
    assert captured["values"] is values
    assert captured["action_logp"] is action_logp
    assert captured["loss_mask"] is loss_mask
    assert captured["rho_bar"] == pytest.approx(2.0)
    assert captured["c_bar"] == pytest.approx(0.5)
    assert captured["float_target"] is float_target
    assert captured["resolve_bootstrap_value"] is resolve_bootstrap_value
    assert captured["batch_value"] is batch_value
    assert batch_value_calls == [(batch, "vtrace_rho_bar"), (batch, "vtrace_c_bar")]
    torch.testing.assert_close(context["targets"], resolved_vtrace.targets)
    torch.testing.assert_close(context["advantages"], resolved_vtrace.advantages)
    torch.testing.assert_close(context["vtrace_rhos"], resolved_vtrace.rhos_for_metrics)
    torch.testing.assert_close(context["rewards"], resolved_vtrace.rewards_for_metrics)
    torch.testing.assert_close(context["policy_train_mask"], loss_mask)
    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False


def test_compute_impala_objective_stage_preserves_context_and_objective_contract() -> None:
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        trajectory_retention_coef=0.5,
        value_loss_coef=0.25,
        entropy_coef=0.1,
    )
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        obs=obs,
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        values=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        trajectory_retention_valid=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        factorized_result=SimpleNamespace(top_action_ids=torch.tensor([[0], [0]], dtype=torch.long)),
        context=context,
    )
    resolved_vtrace = SimpleNamespace(
        advantages=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        targets=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
    )
    policy_action_logp = torch.tensor([[-0.25], [-0.75]], dtype=torch.float32)
    retention_action_logp = torch.tensor([[-0.5], [-1.0]], dtype=torch.float32)
    entropy = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

    stage = compute_impala_objective_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        entropy=entropy,
        resolved_vtrace=resolved_vtrace,
        batch_value=lambda source, key: source.get(key),
    )
    direct = compute_impala_objective_losses(
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        actions=inputs.actions,
        advantages=resolved_vtrace.advantages,
        values=inputs.values,
        targets=resolved_vtrace.targets,
        entropy=entropy,
        loss_mask=inputs.loss_mask,
        value_loss_mask=context["value_train_mask"],
        value_loss_coef=float(learner.value_loss_coef),
        entropy_coef=float(learner.entropy_coef),
        trajectory_retention_valid=inputs.trajectory_retention_valid,
        trajectory_retention_coef=float(learner.trajectory_retention_coef),
        top_action_ids=inputs.factorized_result.top_action_ids,
    )

    torch.testing.assert_close(stage.losses.total_loss, direct.total_loss)
    torch.testing.assert_close(stage.losses.policy_loss, direct.policy_loss)
    torch.testing.assert_close(stage.losses.value_loss, direct.value_loss)
    torch.testing.assert_close(stage.losses.entropy_mean, direct.entropy_mean)
    torch.testing.assert_close(stage.losses.trajectory_retention_loss, direct.trajectory_retention_loss)
    assert stage.losses.trajectory_retention_metrics == pytest.approx(direct.trajectory_retention_metrics)
    assert context["value_train_mask"].requires_grad is False
    assert context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert context["trajectory_retention_loss"].requires_grad is False


def test_apply_impala_policy_anchor_stage_preserves_inputs_loss_and_metrics() -> None:
    anchor_loss = torch.tensor(0.75, dtype=torch.float32)
    calls: list[dict[str, Any]] = []
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_model = object()
    reset_before_step = torch.tensor([[False], [True]], dtype=torch.bool)
    inputs = SimpleNamespace(
        obs=obs,
        loss_mask=loss_mask,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_model=forward_model,
        reset_before_step=reset_before_step,
    )
    batch: dict[str, bool] = {"policy_anchor_batch": True}

    def fake_policy_anchor_loss_and_metrics(
        source_batch: Any,
        *,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
        factorized_result: Any,
        forward_model: Any,
        reset_before_step: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        calls.append(
            {
                "batch": source_batch,
                "obs": obs,
                "loss_mask": loss_mask,
                "packed_legal": packed_legal,
                "factorized_result": factorized_result,
                "forward_model": forward_model,
                "reset_before_step": reset_before_step,
            }
        )
        return anchor_loss, {"policy_anchor_weighted_loss": float(anchor_loss)}

    learner = SimpleNamespace(_policy_anchor_loss_and_metrics=fake_policy_anchor_loss_and_metrics)
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss + anchor_loss)
    assert result.policy_anchor_loss is anchor_loss
    assert result.policy_anchor_metrics["policy_anchor_weighted_loss"] == pytest.approx(0.75)
    assert calls == [
        {
            "batch": batch,
            "obs": obs,
            "loss_mask": loss_mask,
            "packed_legal": packed_legal,
            "factorized_result": factorized_result,
            "forward_model": forward_model,
            "reset_before_step": reset_before_step,
        }
    ]


def test_apply_impala_policy_anchor_stage_preserves_total_loss_when_anchor_disabled() -> None:
    learner = SimpleNamespace(
        _policy_anchor_loss_and_metrics=lambda *args, **kwargs: (
            None,
            {"policy_anchor_disabled": 1.0},
        )
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        packed_legal=None,
        factorized_result=None,
        forward_model=object(),
        reset_before_step=None,
    )
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch={},
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss)
    assert result.policy_anchor_loss is None
    assert result.policy_anchor_metrics == {"policy_anchor_disabled": 1.0}


def test_compute_impala_loss_core_finalizes_vtrace_objective_context_and_metrics() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.25,
        value_loss_coef=1.0,
        entropy_coef=0.0,
    )
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["value_train_mask"] = np.asarray([[False], [True]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    reductions = resolve_impala_action_reductions(
        factorized_result=inputs.factorized_result,
        logits=inputs.logits,
        packed_logits=inputs.packed_logits,
        legal_mask=inputs.legal_mask,
        packed_legal=inputs.packed_legal,
        actions=inputs.actions,
        entropy_scope=learner.entropy_scope,
        pass_action_id=learner.pass_action_id,
        action_catalog=getattr(learner.model, "action_catalog", None),
        record_timing_ms=learner._record_timing_ms,
    )
    inputs.context["action_logp"] = reductions.action_logp.detach()
    inputs.context["entropy"] = reductions.entropy.detach()

    result = compute_impala_loss_core(
        learner=learner,
        batch=batch,
        inputs=inputs,
        action_logp=reductions.action_logp,
        entropy=reductions.entropy,
        batch_value=lambda source, key: source.get(key),
    )

    assert result.context is inputs.context
    assert "targets" in result.context
    assert "advantages" in result.context
    assert "vtrace_rhos" in result.context
    assert "rewards" in result.context
    assert "trajectory_retention_loss" in result.context
    assert result.context["policy_train_mask"].tolist() == [[1.0], [0.0]]
    assert result.context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert result.metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["value_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert result.metrics["trajectory_retention_weighted_loss"] > 0.0
    assert result.metrics["loss"] == pytest.approx(float(result.total_loss.detach()))


def test_compute_impala_loss_pipeline_records_action_reductions_and_core_context() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()

    loss, metrics, context = compute_impala_loss_and_metrics_with_context(
        learner=learner,
        batch=batch,
        batch_value=lambda source, key: source.get(key),
    )

    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert context["action_logp"].shape == torch.Size((2, 1))
    assert context["entropy"].shape == torch.Size((2, 1))
    assert "targets" in context
    assert "advantages" in context
    assert "vtrace_rhos" in context
    assert "policy_train_mask" in context
    assert not context["action_logp"].requires_grad
    assert not context["entropy"].requires_grad
    assert metrics["policy_train_fraction"] == pytest.approx(1.0)


def test_summarize_structured_policy_metrics_reports_mainmove_pressure() -> None:
    action_catalog = _structured_metric_catalog()
    main_move_02_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (
            action_catalog.decode(action_id).family == "main_move"
            and action_catalog.decode(action_id).from_slot == 0
            and action_catalog.decode(action_id).to_slot == 2
        )
    )
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)

    legal_mask[0, 0, [0, main_move_02_action, 25]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, main_move_02_action] = 2.0
    logits[0, 0, 25] = 1.0

    legal_mask[1, 0, [0, main_move_02_action, 25]] = True
    logits[1, 0, 0] = 3.0
    logits[1, 0, main_move_02_action] = 0.0
    logits[1, 0, 25] = 1.0

    metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)

    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(0.5)
    assert 0.0 < metrics["structured_main_move_share_when_play_available"] < 1.0
    assert (
        metrics["structured_main_play_character_mass"]
        + metrics["structured_main_move_mass"]
        + metrics["structured_pass_mass"]
    ) == pytest.approx(1.0)
    assert 0.0 < metrics["structured_exact_action_concentration"] <= 1.0


def test_summarize_structured_policy_metrics_matches_packed_meta_path() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    packed_metrics = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_impala_structured_policy_summary_resolves_dense_mask_when_packed_meta_missing() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = object()

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=26,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    assert resolver_calls == [(batch, torch.Size((2, 1)), 26)]
    assert metrics == pytest.approx(
        summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]
    assert timings[0][1] >= 0.0


def test_compute_impala_structured_policy_summary_keeps_packed_meta_path_without_dense_mask() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                torch.as_tensor(packed_meta, dtype=torch.long),
            ),
            resolve_legal_mask=lambda _source_batch, _expected_shape, _action_dim: pytest.fail(
                "packed metadata path should not reconstruct a dense mask"
            ),
        ),
        record_timing_ms=lambda _name, _duration: None,
    )

    expected = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    assert metrics == pytest.approx(expected)


def test_compute_structured_teacher_auxiliary_metrics_supervises_slot_groups_not_hand_indices() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Row 0: two different hand indices map to the same play slot. Slot supervision should
    # treat their combined probability mass as correct.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1: attack family with the correct attack type.
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        teacher_slot=torch.tensor([[0], [0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [11]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.1,
        attack_type_coef=0.05,
        action_coef=0.15,
        same_family_action_coef=0.2,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_loss"] < 0.05
    assert metrics["teacher_action_loss"] < 0.35
    assert metrics["teacher_same_family_action_loss"] < 0.35


def test_compute_structured_teacher_auxiliary_metrics_groups_main_move_targets_by_destination_slot() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_actions_by_target: dict[int, list[int]] = {}
    for action_id in range(action_catalog.action_space_size):
        decoded = action_catalog.decode(action_id)
        if decoded.family != "main_move" or decoded.to_slot is None:
            continue
        move_actions_by_target.setdefault(int(decoded.to_slot), []).append(int(action_id))
    target_slot, target_actions = next(
        (slot, action_ids) for slot, action_ids in move_actions_by_target.items() if len(action_ids) >= 2
    )
    preferred_move, alternate_move = target_actions[:2]

    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [preferred_move, alternate_move, action_catalog.pass_action_id]] = True
    logits[0, 0, preferred_move] = 1.0
    logits[0, 0, alternate_move] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -4.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        "teacher_slot": torch.tensor([[target_slot]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[preferred_move]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 1.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 1.0,
    }

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)
    assert dense_metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert dense_metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert dense_metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)

    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
    }

    logits[0, 0, 0] = 4.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -5.0
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    logits[0, 0, 0] = 0.5
    logits[0, 0, 5] = 4.0
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_impala_teacher_auxiliary_request_preserves_dense_teacher_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
        profile_timers=True,
    )
    cast(Any, learner)._active_timing_metrics = {}
    expected_shape = torch.Size((1, 1))
    result = compute_impala_teacher_auxiliary(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
        legal_mask=torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool),
        loss_mask=torch.ones(expected_shape, dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=expected_shape,
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        batch_value=lambda batch, key: batch.get(key),
    )

    assert result.loss > 0.0
    assert result.metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.metrics["teacher_family_loss"] > 0.0
    assert result.metrics["teacher_action_loss"] > 0.0
    assert "teacher_family_log_probs" in result.context
    assert cast(Any, learner)._active_timing_metrics["timer_learner_teacher_aux_ms"] >= 0.0


def test_resolve_impala_teacher_auxiliary_labels_preserves_time_major_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    expected_shape = torch.Size((1, 2))
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"], family_index["attack"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0, 1]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1, 0]], dtype=np.int64),
        "teacher_action": np.asarray([[0, 11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True, False]], dtype=np.bool_),
    }

    labels = resolve_impala_teacher_auxiliary_labels(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=expected_shape,
    )

    assert labels.family is not None
    assert labels.family.dtype == torch.long
    assert labels.family.shape == expected_shape
    assert labels.family.tolist() == [[family_index["main_play_character"], family_index["attack"]]]
    assert labels.slot is not None
    assert labels.slot.tolist() == [[0, 1]]
    assert labels.move_source is None
    assert labels.attack_type is not None
    assert labels.attack_type.tolist() == [[-1, 0]]
    assert labels.action is not None
    assert labels.action.tolist() == [[0, 11]]
    assert labels.valid is not None
    assert labels.valid.dtype == torch.bool
    assert labels.valid.tolist() == [[True, False]]


def test_resolve_impala_teacher_auxiliary_coefficients_names_all_teacher_knobs() -> None:
    learner = ImpalaLearner(
        teacher_family_coef=0.1,
        teacher_slot_coef=0.2,
        teacher_hand_coef=0.3,
        teacher_move_source_coef=0.4,
        teacher_attack_type_coef=0.5,
        teacher_action_coef=0.6,
        teacher_same_family_action_coef=0.7,
        teacher_action_margin_coef=0.8,
        teacher_action_margin=0.9,
        teacher_same_family_action_margin_coef=1.1,
        teacher_same_family_action_margin=1.2,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=1.3,
        teacher_public_heuristic_temperature=1.4,
        teacher_public_nonpass_over_pass_coef=1.5,
        teacher_public_nonpass_over_pass_margin=1.6,
        teacher_public_heuristic_families=("main_play_character",),
    )

    coefficients = resolve_impala_teacher_auxiliary_coefficients(learner)

    assert coefficients.family == pytest.approx(0.1)
    assert coefficients.slot == pytest.approx(0.2)
    assert coefficients.hand == pytest.approx(0.3)
    assert coefficients.move_source == pytest.approx(0.4)
    assert coefficients.attack_type == pytest.approx(0.5)
    assert coefficients.action == pytest.approx(0.6)
    assert coefficients.same_family_action == pytest.approx(0.7)
    assert coefficients.action_margin == pytest.approx(0.8)
    assert coefficients.action_margin_value == pytest.approx(0.9)
    assert coefficients.same_family_action_margin == pytest.approx(1.1)
    assert coefficients.same_family_action_margin_value == pytest.approx(1.2)
    assert coefficients.exact_action_families == ("attack",)
    assert coefficients.public_heuristic == pytest.approx(1.3)
    assert coefficients.public_heuristic_temperature == pytest.approx(1.4)
    assert coefficients.public_nonpass_over_pass == pytest.approx(1.5)
    assert coefficients.public_nonpass_over_pass_margin == pytest.approx(1.6)
    assert coefficients.public_heuristic_families == ("main_play_character",)


def test_resolve_impala_teacher_auxiliary_factorized_inputs_preserves_required_and_optional_fields() -> None:
    required = {
        "family_log_probs": torch.zeros((1, 1, 2)),
        "play_slot_log_probs": torch.ones((1, 1, 3)),
        "move_slot_log_probs": torch.full((1, 1, 4), 2.0),
        "attack_slot_log_probs": torch.full((1, 1, 5), 3.0),
        "attack_type_log_probs": torch.full((1, 1, 6), 4.0),
    }
    result = resolve_impala_teacher_auxiliary_factorized_inputs(SimpleNamespace(**required))

    assert result.family_log_probs is required["family_log_probs"]
    assert result.play_slot_log_probs is required["play_slot_log_probs"]
    assert result.move_source_log_probs is None
    assert result.move_slot_log_probs is required["move_slot_log_probs"]
    assert result.attack_slot_log_probs is required["attack_slot_log_probs"]
    assert result.attack_type_log_probs is required["attack_type_log_probs"]
    assert result.top_action_ids is None
    assert result.same_family_action_logp is None
    assert result.same_family_top_action_ids is None


def test_resolve_impala_teacher_auxiliary_packed_inputs_preserves_tuple_contract() -> None:
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()

    packed = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
    )
    dense = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=None,
        packed_view=packed_view,
    )

    assert packed.ids is ids
    assert packed.offsets is offsets
    assert packed.meta is meta
    assert packed.view is packed_view
    assert dense.ids is None
    assert dense.offsets is None
    assert dense.meta is None
    assert dense.view is packed_view


def test_resolve_impala_teacher_auxiliary_inputs_preserves_aggregate_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.25,
        teacher_public_heuristic_coef=0.75,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        same_family_action_logp=torch.tensor([[-0.5]]),
    )
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }

    result = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )

    assert result.labels.family is not None
    assert result.labels.family.tolist() == [[family_index["main_play_character"]]]
    assert result.labels.move_source is None
    assert result.coefficients.family == pytest.approx(0.25)
    assert result.coefficients.public_heuristic == pytest.approx(0.75)
    assert result.coefficients.public_heuristic_families == ("main_play_character",)
    assert result.packed.ids is ids
    assert result.packed.offsets is offsets
    assert result.packed.meta is meta
    assert result.packed.view is packed_view
    assert result.factorized.family_log_probs is factorized_result.family_log_probs
    assert result.factorized.play_slot_log_probs is factorized_result.play_slot_log_probs
    assert result.factorized.same_family_action_logp is factorized_result.same_family_action_logp
    assert result.factorized.same_family_top_action_ids is None


def test_compute_structured_teacher_auxiliary_from_impala_inputs_maps_all_fields(monkeypatch) -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.11,
        teacher_slot_coef=0.12,
        teacher_hand_coef=0.13,
        teacher_move_source_coef=0.14,
        teacher_attack_type_coef=0.15,
        teacher_action_coef=0.16,
        teacher_same_family_action_coef=0.17,
        teacher_action_margin_coef=0.18,
        teacher_action_margin=0.19,
        teacher_same_family_action_margin_coef=0.20,
        teacher_same_family_action_margin=0.21,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=0.22,
        teacher_public_heuristic_temperature=0.23,
        teacher_public_nonpass_over_pass_coef=0.24,
        teacher_public_nonpass_over_pass_margin=0.25,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_source_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 1.5),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        top_action_ids=torch.tensor([[0]], dtype=torch.long),
        same_family_action_logp=torch.tensor([[-0.5]]),
        same_family_top_action_ids=torch.tensor([[5]], dtype=torch.long),
        same_family_arg0_logp=torch.tensor([[-0.6]]),
        same_family_top_arg0=torch.tensor([[1]], dtype=torch.long),
    )
    inputs = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_move_source": np.asarray([[1]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    public_target = torch.arange(2, dtype=torch.float32)
    sentinel_loss = torch.tensor(9.0)
    sentinel_metrics = {"teacher_aux_loss": 9.0}
    sentinel_context = {"teacher_family_log_probs": torch.tensor([1.0])}
    captured: dict[str, Any] = {}

    def fake_compute_structured_teacher_auxiliary_metrics(
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        captured.update(kwargs)
        return sentinel_loss, sentinel_metrics, sentinel_context

    monkeypatch.setattr(
        impala_teacher_auxiliary_call,
        "compute_structured_teacher_auxiliary_metrics",
        fake_compute_structured_teacher_auxiliary_metrics,
    )

    loss, metrics, context = impala_teacher_auxiliary_call.compute_structured_teacher_auxiliary_from_impala_inputs(
        inputs=inputs,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        public_heuristic_target_logits=public_target,
    )

    assert loss is sentinel_loss
    assert metrics is sentinel_metrics
    assert context is sentinel_context
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["teacher_family"] is inputs.labels.family
    assert captured["teacher_slot"] is inputs.labels.slot
    assert captured["teacher_move_source"] is inputs.labels.move_source
    assert captured["teacher_attack_type"] is inputs.labels.attack_type
    assert captured["teacher_action"] is inputs.labels.action
    assert captured["teacher_valid"] is inputs.labels.valid
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] is action_catalog
    assert captured["family_coef"] == pytest.approx(0.11)
    assert captured["slot_coef"] == pytest.approx(0.12)
    assert captured["hand_coef"] == pytest.approx(0.13)
    assert captured["move_source_coef"] == pytest.approx(0.14)
    assert captured["attack_type_coef"] == pytest.approx(0.15)
    assert captured["action_coef"] == pytest.approx(0.16)
    assert captured["same_family_action_coef"] == pytest.approx(0.17)
    assert captured["action_margin_coef"] == pytest.approx(0.18)
    assert captured["action_margin"] == pytest.approx(0.19)
    assert captured["same_family_action_margin_coef"] == pytest.approx(0.20)
    assert captured["same_family_action_margin"] == pytest.approx(0.21)
    assert captured["exact_action_families"] == ("attack",)
    assert captured["public_heuristic_coef"] == pytest.approx(0.22)
    assert captured["public_heuristic_temperature"] == pytest.approx(0.23)
    assert captured["public_nonpass_over_pass_coef"] == pytest.approx(0.24)
    assert captured["public_nonpass_over_pass_margin"] == pytest.approx(0.25)
    assert captured["public_heuristic_families"] == ("main_play_character",)
    assert captured["public_heuristic_target_logits"] is public_target
    assert captured["packed_ids"] is ids
    assert captured["packed_offsets"] is offsets
    assert captured["packed_meta"] is meta
    assert captured["packed_view"] is packed_view
    assert captured["factorized_family_log_probs"] is factorized_result.family_log_probs
    assert captured["factorized_play_slot_log_probs"] is factorized_result.play_slot_log_probs
    assert captured["factorized_move_source_log_probs"] is factorized_result.move_source_log_probs
    assert captured["factorized_move_slot_log_probs"] is factorized_result.move_slot_log_probs
    assert captured["factorized_attack_slot_log_probs"] is factorized_result.attack_slot_log_probs
    assert captured["factorized_attack_type_log_probs"] is factorized_result.attack_type_log_probs
    assert captured["factorized_top_action_ids"] is factorized_result.top_action_ids
    assert captured["factorized_same_family_action_logp"] is factorized_result.same_family_action_logp
    assert captured["factorized_same_family_top_action_ids"] is factorized_result.same_family_top_action_ids
    assert captured["factorized_same_family_arg0_logp"] is factorized_result.same_family_arg0_logp
    assert captured["factorized_same_family_top_arg0"] is factorized_result.same_family_top_arg0


def test_resolve_structured_teacher_zero_context_uses_packed_view_before_loss_mask() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([1.0, 2.0], dtype=torch.float64),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    packed_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=packed_view,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )
    mask_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )

    assert packed_zero.value_dtype == packed_view.logits.dtype
    assert packed_zero.zero.dtype == packed_view.logits.dtype
    assert packed_zero.empty_metrics["teacher_aux_loss"] == pytest.approx(0.0)
    assert mask_zero.value_dtype == torch.float64


def test_resolve_structured_teacher_required_labels_names_missing_label_gate() -> None:
    family = torch.tensor([[0]], dtype=torch.long)
    slot = torch.tensor([[1]], dtype=torch.long)
    attack_type = torch.tensor([[-1]], dtype=torch.long)
    valid = torch.tensor([[True]], dtype=torch.bool)

    labels = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=slot,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )
    missing = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=None,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )

    assert labels is not None
    assert labels.family is family
    assert labels.slot is slot
    assert labels.attack_type is attack_type
    assert labels.valid is valid
    assert missing is None


def test_resolve_structured_teacher_branch_prioritizes_factorized_then_packed_then_dense() -> None:
    factorized = resolve_structured_teacher_branch(
        factorized_family_log_probs=torch.zeros((1, 1, 2)),
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    packed = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    dense = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    inactive = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=None,
    )

    assert factorized.use_factorized is True
    assert factorized.use_packed is False
    assert factorized.use_dense is False
    assert packed.use_factorized is False
    assert packed.use_packed is True
    assert packed.use_dense is False
    assert dense.use_factorized is False
    assert dense.use_packed is False
    assert dense.use_dense is True
    assert inactive.use_factorized is False
    assert inactive.use_packed is False
    assert inactive.use_dense is False


def test_resolve_structured_teacher_dispatch_preserves_label_gate_before_packed_view_build() -> None:
    action_catalog = _teacher_aux_catalog()
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float64)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    invalid_packed_meta = torch.zeros((2, 3), dtype=torch.long)
    labels = {
        "teacher_family": torch.tensor([[0]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
    }

    missing_label_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=invalid_packed_meta,
        packed_view=None,
        factorized_family_log_probs=None,
        teacher_family=labels["teacher_family"],
        teacher_slot=None,
        teacher_attack_type=labels["teacher_attack_type"],
        teacher_valid=labels["teacher_valid"],
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
    )

    assert missing_label_dispatch.labels is None
    assert missing_label_dispatch.packed_view is None
    assert missing_label_dispatch.branch.use_factorized is False
    assert missing_label_dispatch.branch.use_packed is False
    assert missing_label_dispatch.branch.use_dense is False
    assert missing_label_dispatch.zero_context.value_dtype == torch.float64

    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    factorized_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=None,
        factorized_family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        **labels,
    )

    assert factorized_dispatch.labels is not None
    assert factorized_dispatch.packed_view is not None
    assert factorized_dispatch.branch.use_factorized is True
    assert factorized_dispatch.branch.use_packed is False
    assert factorized_dispatch.branch.use_dense is False
    assert factorized_dispatch.zero_context.value_dtype == torch.float64


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    teacher_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    misaligned_view = _packed_structured_legal_view(
        logits=torch.tensor([4.0, 0.5, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=misaligned_view,
        **cast(Any, teacher_kwargs),
    )

    aligned_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=aligned_view,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_gates_public_heuristic_by_family() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    common_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "packed_view": packed_view,
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    allowed_loss, allowed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("main_play_character",),
        **cast(Any, common_kwargs),
    )
    gated_loss, gated_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("attack",),
        **cast(Any, common_kwargs),
    )

    assert float(allowed_loss.detach()) > 0.0
    assert allowed_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert allowed_metrics["teacher_public_heuristic_loss"] > 0.0
    assert float(gated_loss.detach()) == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_loss"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_matches_packed_meta_path() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0
    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_packed_structured_teacher_auxiliary_metrics_matches_dispatcher() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_meta_tensor = torch.as_tensor(packed_meta, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
    )
    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[5]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.0,
        "action_coef": 0.3,
        "same_family_action_coef": 0.4,
    }

    dispatch_loss, dispatch_metrics, dispatch_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
        packed_view=packed_view,
        **cast(Any, teacher_kwargs),
    )
    direct_loss, direct_metrics, direct_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_move_source=None,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(direct_loss, dispatch_loss)
    assert direct_metrics == pytest.approx(dispatch_metrics)
    assert direct_context.keys() == dispatch_context.keys()
    for key in direct_context:
        torch.testing.assert_close(direct_context[key], dispatch_context[key])


def test_compute_packed_teacher_action_supervision_matches_packed_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[5]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    direct = compute_packed_teacher_action_supervision(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_catalog=action_catalog,
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=1.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )

    torch.testing.assert_close(packed_loss, direct.action_loss + direct.same_family_action_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_action_supervision_matches_factorized_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -3.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_log_probs = torch.log_softmax(family_logits, dim=-1)
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[0], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    same_family_logp = torch.tensor([[-0.1], [-0.4]], dtype=torch.float32)
    same_family_top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    zero = family_log_probs.sum() * 0.0

    direct = compute_factorized_teacher_action_supervision(
        family_log_probs=family_log_probs.reshape(-1, family_log_probs.shape[-1]),
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=zero,
        value_dtype=family_log_probs.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.3,
        same_family_action_coef=0.7,
        factorized_family_log_probs=family_log_probs,
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
    )
    expected_action_loss = direct.action_loss * 0.3 + direct.same_family_action_loss * 0.7

    torch.testing.assert_close(factorized_loss, expected_action_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)


def test_compute_packed_teacher_group_supervision_matches_packed_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((3, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((3, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[1, 0, [10, 11, 12, action_catalog.pass_action_id]] = True
    logits[1, 0, 10] = -2.0
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = -1.0
    logits[1, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[2, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[2, 0, move_action] = 3.5
    logits[2, 0, competing_move_action] = -0.5
    logits[2, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["attack"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [0], [int(move_decoded.to_slot or 0)]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [attack_type_index["direct"]], [-1]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [11], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    teacher_move_source = torch.tensor([[-1], [-1], [int(move_decoded.from_slot or 0)]], dtype=torch.long)
    loss_mask = torch.ones((3, 1), dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)

    direct = compute_packed_teacher_group_supervision(
        packed_view=packed_view,
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=teacher_move_source.reshape(-1),
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        action_catalog=action_catalog,
        family_names=metadata.family_names,
        family_index={name: index for index, name in enumerate(metadata.family_names)},
        attack_type_names=metadata.attack_type_names,
        move_source_targets_by_action=None,
        move_source_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=teacher_move_source,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.5,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(packed_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_group_supervision_matches_factorized_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    row_count = 3
    family_logits = torch.full((row_count, 1, len(action_catalog.families)), -4.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_logits[2, 0, family_index["attack"]] = 3.0
    play_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    play_slot_logits[0, 0, 0] = 3.0
    move_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_slot_logits[1, 0, int(move_decoded.to_slot or 0)] = 3.0
    move_source_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_source_logits[1, 0, int(move_decoded.from_slot or 0)] = 3.0
    attack_slot_logits = torch.zeros((row_count, 1, int(action_catalog.attack_slot_count)), dtype=torch.float32)
    attack_type_logits = torch.full((row_count, 1, len(action_catalog.attack_type_names)), -4.0)
    attack_type_logits[2, 0, attack_type_index["direct"]] = 3.0
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]], [family_index["attack"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [int(move_decoded.to_slot or 0)], [0]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [-1], [attack_type_index["direct"]]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [move_action], [10]], dtype=torch.long)
    teacher_valid = torch.ones((row_count, 1), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5], [0.25]], dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)
    zero = family_logits.sum() * 0.0

    direct = compute_factorized_teacher_group_supervision(
        family_log_probs=torch.log_softmax(family_logits, dim=-1).reshape(row_count, -1),
        play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1).reshape(row_count, -1),
        move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1).reshape(row_count, -1),
        move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1).reshape(row_count, -1),
        attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1).reshape(row_count, -1),
        attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1).reshape(row_count, -1),
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=None,
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        attack_type_names=tuple(action_catalog.attack_type_names),
        move_source_targets_by_action=torch.as_tensor(metadata.move_from_slots, dtype=torch.long),
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        attack_family_id=family_index["attack"],
        move_source_coef=1.0,
        zero=zero,
        value_dtype=family_logits.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.5,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1),
        factorized_move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1),
        factorized_move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1),
        factorized_attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1),
        factorized_attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(factorized_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)


def test_compute_packed_teacher_public_supervision_matches_packed_branch_public_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, -0.5, 3.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    target_logits = torch.tensor([4.0, 5.0, -5.0], dtype=torch.float32)

    direct = compute_packed_teacher_public_supervision(
        packed_view=packed_view,
        public_heuristic_target_logits=target_logits,
        public_heuristic_family_ids=(family_index["main_play_character"],),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        pass_action_id=action_catalog.pass_action_id,
        public_heuristic_coef=1.0,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=1.0,
        public_nonpass_over_pass_margin=0.5,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=target_logits,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_public_loss = direct.public_heuristic_loss * 0.7 + direct.public_nonpass_over_pass_loss * 0.3

    torch.testing.assert_close(packed_loss, expected_public_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_packed_teacher_margin_supervision_matches_packed_branch_margin_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[5]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    direct = compute_packed_teacher_margin_supervision(
        packed_view=packed_view,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        action_margin_coef=1.0,
        action_margin=0.5,
        same_family_action_margin_coef=1.0,
        same_family_action_margin=0.5,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.25,
        action_margin=0.5,
        same_family_action_margin_coef=0.75,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_margin_loss = direct.action_margin_loss * 0.25 + direct.same_family_action_margin_loss * 0.75

    torch.testing.assert_close(packed_loss, expected_margin_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_factorized_structured_teacher_reuses_packed_public_and_margin_helpers() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[0]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -3.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    public_target_logits = torch.tensor([4.0, 5.0, -5.0], dtype=torch.float32)
    zero = packed_view.logits.sum() * 0.0
    margin_direct = compute_packed_teacher_margin_supervision(
        packed_view=packed_view,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        action_margin_coef=1.0,
        action_margin=0.5,
        same_family_action_margin_coef=1.0,
        same_family_action_margin=0.5,
        zero=zero,
        value_dtype=packed_view.logits.dtype,
    )
    public_direct = compute_packed_teacher_public_supervision(
        packed_view=packed_view,
        public_heuristic_target_logits=public_target_logits,
        public_heuristic_family_ids=(family_index["main_play_character"],),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        pass_action_id=action_catalog.pass_action_id,
        public_heuristic_coef=1.0,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=1.0,
        public_nonpass_over_pass_margin=0.5,
        zero=zero,
        value_dtype=packed_view.logits.dtype,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.2,
        action_margin=0.5,
        same_family_action_margin_coef=0.4,
        same_family_action_margin=0.5,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=public_target_logits,
        packed_view=packed_view,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )
    expected_loss = (
        margin_direct.action_margin_loss * 0.2
        + margin_direct.same_family_action_margin_loss * 0.4
        + public_direct.public_heuristic_loss * 0.7
        + public_direct.public_nonpass_over_pass_loss * 0.3
    )

    torch.testing.assert_close(aux_loss, expected_loss)
    expected_metrics = {**margin_direct.metrics, **public_direct.metrics}
    for key, value in expected_metrics.items():
        assert metrics[key] == pytest.approx(value)
    expected_context = {**margin_direct.context, **public_direct.context}
    for key, value in expected_context.items():
        torch.testing.assert_close(context[key], value)


def test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[0, 0, move_action] = 4.0
    logits[0, 0, competing_move_action] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_skips_unsupported_packed_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1 carries attack teacher labels but only exposes pass legally, which previously
    # produced NaNs in the packed grouped-log-prob path.
    legal_mask[1, 0, [19]] = True
    logits[1, 0, 19] = 1.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    packed_loss, packed_metrics, packed_context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    assert torch.isfinite(packed_loss)
    assert np.isfinite(packed_metrics["teacher_aux_loss"])
    assert np.isfinite(packed_metrics["teacher_family_loss"])
    assert np.isfinite(packed_metrics["teacher_slot_loss"])
    assert np.isfinite(packed_metrics["teacher_attack_type_loss"])
    assert np.isfinite(packed_metrics["teacher_action_loss"])
    assert np.isfinite(packed_metrics["teacher_same_family_action_loss"])
    assert packed_metrics["teacher_action_supported_fraction"] == pytest.approx(0.5)
    assert packed_metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(0.5)
    assert "teacher_attack_type_log_probs" not in packed_context
    assert "teacher_family_log_probs" in packed_context
    assert "teacher_action_log_probs" in packed_context
    assert "teacher_same_family_action_log_probs" in packed_context
    assert not torch.isnan(packed_context["teacher_family_log_probs"]).any()


def test_compute_structured_teacher_auxiliary_metrics_reports_within_family_tactical_miss() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Two play-character actions share the same play slot. The model picks the wrong hand index,
    # so family and slot stay correct while the exact within-family choice is wrong.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 1.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, 19] = -4.0

    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
    )

    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_same_family_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["main_move"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_action_logp=torch.tensor([[-0.1], [-0.2]], dtype=torch.float32),
        factorized_same_family_top_action_ids=torch.tensor([[0], [move_action]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(1.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_exact_action_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_top_action_ids=torch.tensor([[0]], dtype=torch.long),
        factorized_same_family_action_logp=torch.tensor([[-0.1]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_loss"] > 0.0


def test_compute_factorized_teacher_hand_supervision_matches_factorized_branch_hand_terms() -> None:
    action_catalog = _teacher_aux_hand_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    play_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_play_character"
        and action_catalog.decode(action_id).hand_index is not None
    )
    clock_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "clock_from_hand"
        and action_catalog.decode(action_id).hand_index is not None
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["clock_from_hand"]] = 3.0
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[play_action], [clock_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    arg0_logp = torch.tensor([[-0.05], [-0.20]], dtype=torch.float32)
    top_arg0 = torch.tensor([[play_hand], [clock_hand]], dtype=torch.long)
    metadata = structured_catalog_metadata(action_catalog)
    zero = family_logits.sum() * 0.0

    direct = compute_factorized_teacher_hand_supervision(
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        hand_targets_by_action=torch.as_tensor(metadata.hand_indices, dtype=torch.long),
        hand_family_ids=(family_index["main_play_character"], family_index["clock_from_hand"]),
        play_family_id=family_index["main_play_character"],
        clock_from_hand_family_id=family_index["clock_from_hand"],
        hand_coef=1.0,
        zero=zero,
        value_dtype=family_logits.dtype,
    )
    factorized_loss, factorized_metrics, _factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=0.4,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
    )

    torch.testing.assert_close(factorized_loss, direct.hand_loss * 0.4)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_hand_targets() -> None:
    action_catalog = _teacher_aux_hand_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    play_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_play_character"
        and action_catalog.decode(action_id).hand_index is not None
    )
    clock_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "clock_from_hand"
        and action_catalog.decode(action_id).hand_index is not None
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["clock_from_hand"]] = 3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[play_action], [clock_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_arg0_logp=torch.tensor([[-0.05], [-0.10]], dtype=torch.float32),
        factorized_same_family_top_arg0=torch.tensor([[play_hand], [clock_hand]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_clock_from_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_hand_loss"] == pytest.approx(0.075)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_move_source_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_explicit_move_source_labels() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[-1]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
        teacher_move_source=torch.tensor([[int(move_decoded.from_slot or 0)]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_reports_family_coverage_on_active_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.zeros((4, 1, len(action_catalog.families)), dtype=torch.float32)
    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [
                [family_index["main_play_character"]],
                [family_index["main_move"]],
                [family_index["attack"]],
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [1], [0], [2]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1], [0], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [5], [11], [5]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True], [True], [False]], dtype=torch.bool),
        loss_mask=torch.tensor([[1.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )

    assert metrics["teacher_active_fraction"] == pytest.approx(0.75)
    assert metrics["teacher_main_play_character_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_main_move_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_attack_fraction"] == pytest.approx(0.0)


def test_impala_learner_auxiliary_update_uses_factorized_same_family_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_same_family_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_hand_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_hand_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)


def test_impala_learner_factorized_policy_anchor_penalizes_post_anchor_drift() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(
        model=model,
        policy_anchor_coef=0.5,
        policy_anchor_temperature=1.0,
    )
    learner._ensure_policy_anchor_model()
    with torch.no_grad():
        model.bias.fill_(2.0)
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["policy_anchor_coef_active"] == pytest.approx(0.5)
    assert metrics["policy_anchor_loss"] > 0.0
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(metrics["policy_anchor_loss"] * 0.5)
    assert metrics["policy_anchor_candidate_count"] == pytest.approx(float(packed_ids.shape[0]))
    assert model.factorized_candidate_logp_calls == 1
    assert learner._policy_anchor_model is not None


def test_impala_learner_reset_policy_anchor_refreshes_current_weights() -> None:
    model = TinyPolicyValueModel()
    learner = ImpalaLearner(model=model, policy_anchor_coef=0.5)
    learner._ensure_policy_anchor_model()

    with torch.no_grad():
        model.policy.bias.fill_(3.0)
    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is not None
    anchor_bias = dict(learner._policy_anchor_model.state_dict())["policy.bias"]
    assert torch.equal(anchor_bias, model.policy.bias.detach())


def test_impala_learner_reset_policy_anchor_clears_disabled_anchor() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    learner._ensure_policy_anchor_model()

    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is None


def test_impala_learner_auxiliary_update_uses_factorized_teacher_action_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_move_source_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_move_source_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([move_action, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_move"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[int(move_decoded.to_slot or 0)]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[move_action]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_public_heuristic_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.trunk_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_loss"] > 0.0
    assert metrics["teacher_public_heuristic_top1_mass"] < 0.1


def test_impala_learner_factorized_margin_aux_uses_factorized_candidate_log_probs_without_public_teacher() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_margin_coef=1.0,
        teacher_action_margin=0.5,
        teacher_same_family_action_margin_coef=1.0,
        teacher_same_family_action_margin=0.5,
        teacher_public_heuristic_coef=0.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 0
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_action_margin_mean"] > 0.5
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_margin_mean"] > 0.5


def test_impala_learner_paired_swing_auxiliary_dense_path_preserves_weighted_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TinyStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = 0.0
        model.policy.bias[5] = 1.0
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "actions": np.asarray([[5]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=_packed_meta_from_ids(action_catalog, packed_ids),
            action_space=action_catalog.action_space_size,
        ),
    }

    loss, metrics, context = learner._paired_swing_loss_and_metrics(
        batch,
        margin=0.25,
        coef=0.5,
        positive_action_source="teacher_action",
        negative_action_source="actions",
    )

    assert loss.detach().item() == pytest.approx(0.625)
    assert metrics["paired_swing_weighted_loss"] == pytest.approx(0.625)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_rows"] == 1.0
    assert context["paired_swing_margins"].tolist() == pytest.approx([-1.0])


def test_compute_paired_swing_candidate_view_preserves_dense_path_outputs() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TinyStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = 0.0
        model.policy.bias[5] = 1.0
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "actions": np.asarray([[5]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=_packed_meta_from_ids(action_catalog, packed_ids),
            action_space=action_catalog.action_space_size,
        ),
    }
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired-swing replay requires packed legal_ids/legal_offsets",
    )

    candidate_view = compute_paired_swing_candidate_view(
        learner,
        batch,
        obs=inputs.obs,
        expected_shape=inputs.expected_shape,
        packed_legal=inputs.packed_legal,
        loss_mask=inputs.loss_mask,
        margin_retention_coef=0.0,
        top_action_retention_coef=0.0,
    )

    assert candidate_view.reference_packed_logits is None
    assert candidate_view.logits is not None
    assert candidate_view.values is not None
    assert candidate_view.zero.item() == pytest.approx(0.0)
    assert candidate_view.packed_view.logits.tolist() == pytest.approx([0.0, 1.0])
    assert candidate_view.logits.shape == torch.Size([1, 1, action_catalog.action_space_size])
    assert candidate_view.values.shape == torch.Size([1, 1])


def test_build_paired_swing_auxiliary_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_swing_auxiliary_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.5,
        margin=0.25,
        positive_action_source="teacher_action",
        negative_action_source="actions",
        loss_scope="label_mean",
        compare_to=" Top_Other ",
        margin_retention_coef=0.1,
        margin_retention_margin=0.2,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        swing_metrics={"paired_swing_rows": 2.0, "paired_swing_weighted_loss": 99.0},
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_swing_weighted_loss"] == 99.0
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_loss_scope_label_mean"] == 1.0
    assert metrics["paired_swing_compare_to_top_other"] == 1.0
    assert metrics["paired_swing_margin_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_swing_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_swing_rows"] == 2.0


def test_resolve_paired_auxiliary_batch_inputs_preserves_default_loss_mask_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyPolicyValueModel(), pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "legal_ids": np.concatenate([packed_ids, packed_ids]),
        "legal_offsets": np.asarray([0, 3, 6], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, np.concatenate([packed_ids, packed_ids])),
    }

    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired helper requires packed legal actions",
    )

    assert inputs.obs.shape == (2, 1, 2)
    assert inputs.expected_shape == torch.Size([2, 1])
    assert inputs.loss_mask.shape == torch.Size([2, 1])
    assert torch.all(inputs.loss_mask == 1.0)
    assert inputs.packed_legal[0].tolist() == [0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id]


def test_resolve_paired_auxiliary_batch_inputs_preserves_missing_packed_error() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    batch = {"obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="paired helper requires packed legal actions"):
        resolve_paired_auxiliary_batch_inputs(
            learner,
            batch,
            packed_legal_error="paired helper requires packed legal actions",
        )


def test_compute_paired_outcome_candidate_logps_preserves_current_reference_views() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired outcome preference replay requires packed legal_ids/legal_offsets",
    )
    actions = learner._require_actions(batch["actions"], expected_shape=inputs.expected_shape)

    candidate_logps = compute_paired_outcome_candidate_logps(
        learner,
        batch,
        obs=inputs.obs,
        packed_legal=inputs.packed_legal,
        actions=actions,
        reset_before_step=None,
    )

    assert candidate_logps.current_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.current_best_non_target_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_best_non_target_logp.shape == torch.Size([2, 1])
    assert torch.allclose(candidate_logps.current_action_logp, candidate_logps.reference_action_logp)
    assert torch.allclose(candidate_logps.current_best_non_target_logp, candidate_logps.reference_best_non_target_logp)
    assert candidate_logps.current_action_logp[0, 0] > candidate_logps.current_best_non_target_logp[0, 0]
    assert candidate_logps.current_action_logp[1, 0] < candidate_logps.current_best_non_target_logp[1, 0]
    assert model.factorized_candidate_logp_calls == 1


def test_build_paired_outcome_preference_context_preserves_detached_logp_surface() -> None:
    current_candidates = torch.tensor([0.0, 1.0], requires_grad=True)
    reference_candidates = torch.tensor([1.0, 0.0], requires_grad=True)
    candidate_logps = PairedOutcomeCandidateLogps(
        current_candidate_log_probs=current_candidates,
        reference_candidate_log_probs=reference_candidates,
        current_action_logp=torch.tensor([[0.1]], requires_grad=True),
        current_best_non_target_logp=torch.tensor([[0.2]], requires_grad=True),
        reference_action_logp=torch.tensor([[0.3]], requires_grad=True),
        reference_best_non_target_logp=torch.tensor([[0.4]], requires_grad=True),
    )

    context = build_paired_outcome_preference_context(
        weighted_loss=torch.tensor(0.75, requires_grad=True),
        loss_mask=torch.tensor([[1.0]], requires_grad=True),
        candidate_logps=candidate_logps,
        preference_context={"paired_outcome_preference_margins": torch.tensor([0.5])},
    )

    assert context["paired_outcome_preference_loss"].item() == pytest.approx(0.75)
    assert context["policy_train_mask"].tolist() == [[1.0]]
    assert context["current_action_logp"].reshape(-1).tolist() == pytest.approx([0.1])
    assert context["current_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.2])
    assert context["reference_action_logp"].reshape(-1).tolist() == pytest.approx([0.3])
    assert context["reference_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.4])
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.5])
    assert not context["paired_outcome_preference_loss"].requires_grad
    assert not context["policy_train_mask"].requires_grad
    assert not context["current_action_logp"].requires_grad


def test_build_paired_outcome_preference_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_outcome_preference_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.7,
        beta=0.2,
        aggregation=" Sum ",
        group_balance=True,
        retention_coef=0.1,
        retention_margin=0.2,
        retention_reference_top_only=True,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        top_action_retention_reference_top_only=True,
        preference_metrics={
            "paired_outcome_preference_pair_count": 2.0,
            "paired_outcome_preference_weighted_loss": 99.0,
        },
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_outcome_preference_weighted_loss"] == 99.0
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_outcome_preference_retention_margin"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_coef"] == pytest.approx(0.3)
    assert metrics["paired_outcome_preference_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_outcome_preference_top_action_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 2.0


def test_impala_learner_paired_outcome_auxiliary_preserves_factorized_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "preference_pair_id": np.asarray([[7], [7]], dtype=np.int64),
        "preference_role": np.asarray([[1], [0]], dtype=np.int64),
        "preference_group_id": np.asarray([[3], [3]], dtype=np.int64),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    loss, metrics, context = learner._paired_outcome_preference_loss_and_metrics(
        batch,
        beta=0.2,
        coef=0.7,
        aggregation="sum",
        group_balance=True,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(metrics["paired_outcome_preference_weighted_loss"])
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 1.0
    assert metrics["paired_outcome_preference_group_count"] == 1.0
    assert context["current_action_logp"].shape == (2, 1)
    assert context["reference_action_logp"].shape == (2, 1)
    assert context["current_best_non_target_logp"].shape == (2, 1)
    assert context["reference_best_non_target_logp"].shape == (2, 1)
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.0])
    assert model.factorized_candidate_logp_calls == 1


def test_impala_learner_auxiliary_update_averages_multiple_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 3
    assert model.public_target_profiles == ["base", "aggressive", "control"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_auxiliary_update_cycles_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["aggressive"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_public_heuristic_profiles_fall_back_to_base_after_end_update() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=0,
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["base"]


def test_impala_learner_mixed_precision_flag_disables_amp_on_cpu() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), mixed_precision=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["loss"] != 0.0
    assert learner._amp_enabled is False
    assert learner._grad_scaler is None


def test_impala_learner_uses_compiled_forward_model_when_provided() -> None:
    base_model = TinyPolicyValueModel(action_dim=2)
    compiled_proxy = ForwardProxyModel(base_model)
    learner = ImpalaLearner(model=base_model, compiled_model=compiled_proxy)

    loss, _metrics = learner._loss_and_metrics(_simple_training_batch())

    assert float(loss.detach()) != 0.0
    assert compiled_proxy.forward_calls == 2


def test_impala_learner_dense_trajectory_retention_is_separate_from_policy_train_mask() -> None:
    torch.manual_seed(0)
    base_model = TinyPolicyValueModel(action_dim=2)
    retention_model = TinyPolicyValueModel(action_dim=2)
    retention_model.load_state_dict(base_model.state_dict())
    base_learner = ImpalaLearner(model=base_model)
    retention_learner = ImpalaLearner(model=retention_model, trajectory_retention_coef=0.4)
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    base_loss, _base_metrics = base_learner._loss_and_metrics(batch)
    retention_loss, retention_metrics = retention_learner._loss_and_metrics(batch)

    assert retention_metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert retention_metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_supported_fraction"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_weighted_loss"] > 0.0
    assert float(retention_loss.detach()) == pytest.approx(
        float(base_loss.detach()) + retention_metrics["trajectory_retention_weighted_loss"]
    )


def test_impala_learner_forward_time_major_matches_manual_legacy_rollout() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=3)
    learner = ImpalaLearner(model=model)
    obs = torch.tensor(
        [
            [[0.25, -0.5], [1.0, 0.0]],
            [[-0.75, 0.5], [0.125, 0.25]],
        ],
        dtype=torch.float32,
    )
    initial_hidden = torch.ones((2, 1), dtype=torch.float32)

    with torch.no_grad():
        learner_logits, learner_values = learner._forward_time_major(obs, initial_hidden_state=initial_hidden)

        manual_hidden = initial_hidden
        manual_logits_steps: list[torch.Tensor] = []
        manual_value_steps: list[torch.Tensor] = []
        for step_obs in obs.unbind(dim=0):
            step_logits, step_value, manual_hidden = model(step_obs, manual_hidden)
            manual_logits_steps.append(step_logits)
            manual_value_steps.append(step_value)

    torch.testing.assert_close(learner_logits, torch.stack(manual_logits_steps, dim=0))
    torch.testing.assert_close(learner_values, torch.stack(manual_value_steps, dim=0))


def test_impala_learner_auxiliary_update_optimizes_teacher_only_loss() -> None:
    torch.manual_seed(0)

    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((2, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    legal_mask[1, 0, [10, 11, 12, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_auxiliary_update_handles_batches_without_valid_teacher_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((1, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[False]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_raw_vtrace_inputs_use_current_policy_logp_for_importance_weights() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": (action_logp - 2.0).cpu().numpy().astype(np.float32),
        "behavior_values": values.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, metrics = learner._loss_and_metrics(raw_batch)

    assert metrics["vtrace_rho_p50"] > 7.0
    assert metrics["vtrace_rho_p95"] > 7.0
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(1.0)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)


def test_impala_learner_raw_vtrace_uses_behavior_logp_on_non_train_rows_dense() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=10.0, vtrace_c_bar=10.0)
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, _values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )
    behavior_logp = action_logp.clone()
    behavior_logp[1, 0] = behavior_logp[1, 0] - 3.0

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    torch.testing.assert_close(context["vtrace_rhos"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(context["vtrace_rhos"][1, 0], torch.tensor(1.0))
    assert context["policy_train_mask"].tolist() == [[1.0], [0.0]]


def test_resolve_impala_vtrace_targets_preserves_off_policy_train_rows_and_masks_non_train_rows() -> None:
    values = torch.zeros((2, 1), dtype=torch.float32)
    action_logp = torch.tensor([[0.0], [-1.0]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-2.0], [-4.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    def float_target(value: Any, *, expected_shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        assert tensor.shape == expected_shape
        return tensor

    def resolve_bootstrap(_batch: Any, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros((batch_size,), dtype=like.dtype, device=like.device)

    resolved = resolve_impala_vtrace_targets(
        batch={
            "rewards": torch.zeros((2, 1), dtype=torch.float32),
            "discounts": torch.ones((2, 1), dtype=torch.float32),
            "behavior_logp": behavior_logp,
        },
        vtrace_result=None,
        values=values,
        action_logp=action_logp,
        loss_mask=loss_mask,
        rho_bar=10.0,
        c_bar=10.0,
        float_target=float_target,
        resolve_bootstrap_value=resolve_bootstrap,
        batch_value=lambda batch, key: batch.get(key),
    )

    torch.testing.assert_close(resolved.action_logp, torch.tensor([[0.0], [-4.0]]))
    torch.testing.assert_close(resolved.behavior_logp_for_mask, behavior_logp)
    assert resolved.rhos_for_metrics[0, 0] == pytest.approx(float(np.exp(2.0)))
    assert resolved.rhos_for_metrics[1, 0] == pytest.approx(1.0)
    assert resolved.targets.requires_grad is False
    assert resolved.advantages.requires_grad is False


def test_impala_learner_trains_value_on_non_policy_rows_by_default() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_train_fraction"] == pytest.approx(1.0)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert context["value_train_mask"].tolist() == [[1.0], [1.0]]


def test_impala_learner_accepts_explicit_value_train_mask() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "value_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["value_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_loss"] == pytest.approx(0.0)
    assert context["value_train_mask"].tolist() == [[1.0], [0.0]]


def test_impala_learner_raw_vtrace_inputs_use_current_learner_values_for_targets() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, -0.5]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)

    with torch.no_grad():
        forward = learner._forward_time_major(torch.from_numpy(obs))
        logits = forward.logits
        assert logits is not None
        values = forward.values
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )

    log_rho = -0.2
    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": (action_logp - log_rho).cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), 123.0, dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 2.4,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    expected_rho = float(np.exp(log_rho))
    expected_targets = values.detach() * (1.0 - expected_rho)
    assert torch.allclose(context["targets"], expected_targets, atol=1.0e-6)


def test_impala_learner_raw_vtrace_inputs_can_bootstrap_from_current_model() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=SeatAwareTinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)
    to_play_seat = np.asarray([[0]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    bootstrap_obs = np.asarray([[2.0, 0.0]], dtype=np.float32)
    bootstrap_actor = np.asarray([1], dtype=np.int64)
    final_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            to_play_seat=to_play_seat,
            initial_hidden_state=initial_hidden_state,
        )
        logits = forward.logits
        assert logits is not None
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )
        model = cast(Any, learner.model)
        expected_bootstrap = model.value_seat_aware(
            torch.from_numpy(bootstrap_obs),
            torch.from_numpy(bootstrap_actor),
            torch.from_numpy(final_hidden_state),
        )

    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": action_logp.cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), -77.0, dtype=np.float32),
        "bootstrap_value": np.full((1,), 123.0, dtype=np.float32),
        "bootstrap_obs": bootstrap_obs,
        "bootstrap_actor": bootstrap_actor,
        "final_hidden_state": final_hidden_state,
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    assert torch.allclose(context["targets"], expected_bootstrap.reshape(1, 1), atol=1.0e-6)


def test_impala_learner_forward_time_major_requires_packed_meta_for_structured_updates() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    legal_actions = LegalActionBatch.from_packed(
        np.asarray([0, 5, 19], dtype=np.uint32),
        np.asarray([0, 3], dtype=np.uint32),
        action_space=action_catalog.action_space_size,
    )

    with pytest.raises(ValueError, match="packed legal_actions metadata"):
        learner._forward_time_major(
            obs,
            to_play_seat=np.asarray([[0]], dtype=np.int64),
            legal_actions=legal_actions,
        )


def test_impala_learner_forward_time_major_uses_sequence_fast_path_and_records_packed_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=SequenceStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, SequenceStructuredTeacherModel)
    assert logits.shape == (2, 1, action_catalog.action_space_size)
    assert values.shape == (2, 1)
    assert model.sequence_calls == 1
    assert model.step_calls == 0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(6.0)
    assert learner._active_timing_metrics["packed_candidate_rows"] == pytest.approx(2.0)
    assert learner._active_timing_metrics["avg_legal_actions_per_row"] == pytest.approx(3.0)
    assert learner._active_timing_metrics["timer_learner_forward_time_major_ms"] >= 0.0


def test_impala_learner_forward_time_major_uses_trunk_sequence_path_and_records_breakdown() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TrunkStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    packed_logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, TrunkStructuredTeacherModel)
    assert packed_logits.shape == (6,)
    assert values.shape == (2, 1)
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.sequence_calls == 0
    assert learner._active_timing_metrics["timer_learner_trunk_ms"] >= 0.0
    assert learner._active_timing_metrics["timer_learner_packed_scorer_ms"] >= 0.0


def test_impala_learner_uses_factorized_legal_policy_path_for_loss_and_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        profile_timers=True,
        trajectory_retention_coef=0.06,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.1], [0.2]], dtype=np.float32),
            pg_advantages=np.asarray([[1.0], [0.5]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    loss, metrics = learner._loss_and_metrics(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert float(loss.detach()) != 0.0
    assert model.factorized_calls == 1
    assert learner._active_timing_metrics["timer_learner_factorized_policy_ms"] >= 0.0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(7.0)
    assert metrics["entropy"] > 0.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(0.25)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(0.015)


def test_impala_learner_restricts_packed_policy_scoring_to_train_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[5], [11]], dtype=np.int64),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=action_catalog.action_space_size,
        ),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": np.asarray([[-2.0], [-3.0]], dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert float(loss.detach()) != 0.0
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.scorer_row_count == 1
    assert model.scorer_candidate_count == 3
    assert learner._active_timing_metrics["packed_candidate_train_rows"] == pytest.approx(1.0)
    assert learner._active_timing_metrics["packed_candidate_train_count"] == pytest.approx(3.0)
    assert float(context["vtrace_rhos"][1, 0]) == pytest.approx(1.0)
    assert float(context["vtrace_rhos"][0, 0]) > 1.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)


def test_impala_learner_packed_raw_vtrace_rho_is_one_when_behavior_matches_policy() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    obs = np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32)
    actions = np.asarray([[5], [11]], dtype=np.int64)
    to_play_seat = np.asarray([[0], [1]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            legal_actions=legal_actions,
        )
        assert forward.packed_logits is not None
        behavior_logp, _entropy = packed_scores_action_logp_and_entropy(
            forward.packed_logits,
            torch.as_tensor(packed_ids, dtype=torch.long),
            torch.as_tensor(packed_offsets, dtype=torch.long),
            torch.from_numpy(actions),
            pass_action_id=action_catalog.pass_action_id,
        )
    learner._active_timing_metrics = {}
    batch = {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    torch.testing.assert_close(context["action_logp"], behavior_logp)
    torch.testing.assert_close(context["vtrace_rhos"], torch.ones_like(context["vtrace_rhos"]))
    assert metrics["target_behavior_logp_delta_abs_p99"] == pytest.approx(0.0)
    assert metrics["target_behavior_train_logp_delta_abs_p99"] == pytest.approx(0.0)


def test_impala_learner_auxiliary_update_uses_factorized_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)


def test_impala_learner_reports_reward_advantage_and_chosen_action_metrics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), pass_action_id=1)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.25], [-0.5]], dtype=np.float32),
            pg_advantages=np.asarray([[1.5], [-0.25]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "rewards": np.asarray([[0.0], [1.0]], dtype=np.float32),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["reward_std"] == pytest.approx(0.5)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["reward_min"] == pytest.approx(0.0)
    assert metrics["reward_max"] == pytest.approx(1.0)
    assert metrics["reward_nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["reward_positive_fraction"] == pytest.approx(0.5)
    assert metrics["reward_negative_fraction"] == pytest.approx(0.0)
    assert metrics["advantage_mean"] == pytest.approx(0.625)
    assert metrics["advantage_abs_mean"] == pytest.approx(0.875)
    assert metrics["target_mean"] == pytest.approx(-0.125)
    assert metrics["target_abs_mean"] == pytest.approx(0.375)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.5)
    assert metrics["chosen_pass_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_pass_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_nonpass_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)


def test_impala_loss_metrics_builder_preserves_training_diagnostics_contract() -> None:
    metrics = build_impala_loss_metrics(
        total_loss=torch.tensor(2.0),
        policy_loss=torch.tensor(0.5),
        value_loss=torch.tensor(1.25),
        entropy_mean=torch.tensor(0.125),
        entropy_scope="family",
        loss_mask=torch.tensor([[1.0], [0.0]]),
        value_loss_mask=torch.tensor([[1.0], [1.0]]),
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        action_logp=torch.tensor([[-0.2], [-0.3]]),
        behavior_logp_for_mask=torch.tensor([[-0.5], [-0.3]]),
        rewards_for_metrics=torch.tensor([[0.0], [1.0]]),
        advantages=torch.tensor([[1.5], [-0.25]]),
        targets=torch.tensor([[0.25], [-0.5]]),
        rhos_for_metrics=torch.tensor([[2.0], [4.0]]),
        rho_bar=3.0,
        c_bar=1.5,
        action_catalog=None,
        pass_action_id=1,
        terminal_outcome_backfill_count=7,
        terminal_outcome_backfill_total_micros=11,
        terminal_outcome_trace_backfill_count=13,
        terminal_outcome_trace_backfill_total_micros=17,
        trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
        policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
        teacher_metrics={"teacher_valid_fraction": 0.5},
    )

    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["vtrace_rho_mean"] == pytest.approx(3.0)
    assert metrics["vtrace_train_rho_mean"] == pytest.approx(2.0)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)
    assert metrics["target_behavior_logp_delta_abs_mean"] == pytest.approx(0.15)
    assert metrics["target_behavior_train_logp_delta_abs_mean"] == pytest.approx(0.3)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(7.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(17.0)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.5)


def test_impala_learner_reports_family_chosen_action_outcome_metrics() -> None:
    catalog = _mulligan_metric_catalog()

    metrics = _chosen_action_outcome_metrics(
        actions=torch.tensor([[0], [1], [3], [5], [8]], dtype=torch.long),
        loss_mask=torch.tensor([[True], [True], [False], [True], [True]]),
        rewards=torch.tensor([[0.0], [1.0], [99.0], [2.0], [3.0]], dtype=torch.float32),
        advantages=torch.tensor([[0.5], [-0.25], [99.0], [1.25], [-1.0]], dtype=torch.float32),
        action_catalog=catalog,
        pass_action_id=catalog.pass_action_id,
    )

    assert metrics["chosen_mulligan_confirm_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_confirm_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_mulligan_confirm_train_advantage_mean"] == pytest.approx(0.5)
    assert metrics["chosen_mulligan_select_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_select_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_mulligan_select_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_attack_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_attack_train_advantage_mean"] == pytest.approx(1.25)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_main_play_character_train_fraction"] == pytest.approx(0.0)


def test_impala_learner_amp_overflow_is_reported_without_raising() -> None:
    learner = ImpalaLearner(model=NaNGradientModel())
    cast(Any, learner)._grad_scaler = FakeGradScaler(overflow=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["amp_grad_overflow"] == 1.0
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert learner.update_count == 1
