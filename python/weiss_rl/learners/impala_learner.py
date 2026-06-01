"""IMPALA learner helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.diagnostics.training_logger import TrainingLogger
from weiss_rl.learners import structured_auxiliary as _structured_auxiliary
from weiss_rl.learners.action_logp import (
    learner_logp_from_legal_ids as _learner_logp_from_legal_ids,
)
from weiss_rl.learners.action_logp import (
    learner_logp_from_mask as _learner_logp_from_mask,
)
from weiss_rl.learners.action_logp import (
    masked_action_logp_and_entropy,
    masked_log_probs_and_entropy,
    packed_action_logp_and_entropy,
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
    packed_selected_action_logp,
    packed_subset_action_logp_and_top_action,
)
from weiss_rl.learners.factorized_evaluation import ImpalaFactorizedEvaluationMixin
from weiss_rl.learners.forward_time_major import ForwardTimeMajorResult, time_step_legal_actions
from weiss_rl.learners.impala_auxiliary_loss import ImpalaAuxiliaryLossMixin
from weiss_rl.learners.impala_loss_metrics import (
    chosen_action_outcome_metrics,
)
from weiss_rl.learners.impala_loss_pipeline import compute_impala_loss_and_metrics_with_context
from weiss_rl.learners.impala_policy_anchor_support import ImpalaPolicyAnchorSupportMixin
from weiss_rl.learners.impala_support import ImpalaSupportMixin
from weiss_rl.learners.impala_update_loop import ImpalaUpdateLoopMixin
from weiss_rl.learners.structured_auxiliary import (
    PackedStructuredLegalView as _PackedStructuredLegalView,
)
from weiss_rl.learners.structured_auxiliary import (
    normalize_public_heuristic_profile_mode,
    normalize_public_heuristic_profiles,
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
    resolve_public_heuristic_family_ids,
    structured_catalog_metadata,
)
from weiss_rl.learners.structured_policy_metrics import (
    summarize_structured_policy_metrics as _summarize_structured_policy_metrics,
)
from weiss_rl.learners.structured_teacher_auxiliary import (
    compute_structured_teacher_auxiliary_metrics as _compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.tensor_ops import (
    nonfinite_indices,
    segment_group_sum,
    segment_logsumexp,
    segment_max,
    weighted_mean,
)
from weiss_rl.learners.update_bookkeeping import (
    learner_acceleration_state,
    record_timing_ms,
    should_emit_structured_metrics,
    teacher_aux_active,
)
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_diagnostics import (
    VTRACE_RHO_PERCENTILES as _VTRACE_RHO_PERCENTILES,
)
from weiss_rl.learners.vtrace_diagnostics import (
    summarize_vtrace_diagnostics as _summarize_vtrace_diagnostics,
)
from weiss_rl.learners.vtrace_torch import compute_vtrace_targets_torch

_SUPPORTED_PUBLIC_HEURISTIC_PROFILES = _structured_auxiliary.SUPPORTED_PUBLIC_HEURISTIC_PROFILES
_SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES = _structured_auxiliary.SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES
VTRACE_RHO_PERCENTILES = _VTRACE_RHO_PERCENTILES
_ForwardTimeMajorResult = ForwardTimeMajorResult
_compute_vtrace_targets_torch = compute_vtrace_targets_torch
_masked_log_probs_and_entropy = masked_log_probs_and_entropy
_nonfinite_indices = nonfinite_indices
_normalize_public_heuristic_profile_mode = normalize_public_heuristic_profile_mode
_normalize_public_heuristic_profiles = normalize_public_heuristic_profiles
_packed_action_logp_and_entropy = packed_action_logp_and_entropy
_packed_scores_family_entropy = packed_scores_family_entropy
_packed_group_log_probs = packed_group_log_probs
_packed_scores_action_logp_and_entropy = packed_scores_action_logp_and_entropy
_packed_soft_target_cross_entropy = packed_soft_target_cross_entropy
_packed_structured_legal_view = packed_structured_legal_view
_packed_subset_action_logp_and_top_action = packed_subset_action_logp_and_top_action
_resolve_public_heuristic_family_ids = resolve_public_heuristic_family_ids
_segment_group_sum = segment_group_sum
_segment_logsumexp = segment_logsumexp
_segment_max = segment_max
_structured_catalog_metadata = structured_catalog_metadata
_time_step_legal_actions = time_step_legal_actions
_weighted_mean = weighted_mean
_chosen_action_outcome_metrics = chosen_action_outcome_metrics
compute_structured_teacher_auxiliary_metrics = _compute_structured_teacher_auxiliary_metrics


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return _learner_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return _learner_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions, pass_action_id=pass_action_id)


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    return _summarize_vtrace_diagnostics(result, rho_bar=rho_bar, c_bar=c_bar)


def summarize_structured_policy_metrics(
    logits: Tensor | None,
    legal_mask: Tensor | None,
    *,
    action_catalog: ActionCatalog,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: _PackedStructuredLegalView | None = None,
    factorized_family_log_probs: Tensor | None = None,
) -> dict[str, float]:
    return _summarize_structured_policy_metrics(
        logits,
        legal_mask,
        action_catalog=action_catalog,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=packed_view,
        factorized_family_log_probs=factorized_family_log_probs,
    )


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    return masked_action_logp_and_entropy(logits, legal_mask, actions, pass_action_id=pass_action_id)


def _packed_selected_action_logp(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
    strict: bool = True,
) -> Tensor:
    return packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
        strict=strict,
    )


@dataclass(slots=True)
class ImpalaLearner(
    ImpalaUpdateLoopMixin,
    ImpalaAuxiliaryLossMixin,
    ImpalaFactorizedEvaluationMixin,
    ImpalaPolicyAnchorSupportMixin,
    ImpalaSupportMixin,
):
    model: nn.Module | None = None
    compiled_model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_scope: str = "candidate"
    grad_norm_clip: float = 40.0
    mixed_precision: bool = False
    checkpoint_dir: Path | None = None
    fault_dir: Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: Path | None = None
    logging_interval_updates: int = 100
    vtrace_rho_bar: float = 2.4
    vtrace_c_bar: float = 1.0
    pass_action_id: int | None = None
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_action_margin_coef: float = 0.0
    teacher_action_margin: float = 0.5
    teacher_same_family_action_margin_coef: float = 0.0
    teacher_same_family_action_margin: float = 0.5
    teacher_exact_action_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_nonpass_over_pass_coef: float = 0.0
    teacher_public_nonpass_over_pass_margin: float = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_anchor_coef: float = 0.0
    policy_anchor_top_action_coef: float = 0.0
    policy_anchor_temperature: float = 1.0
    trajectory_retention_coef: float = 0.0
    profile_timers: bool = False
    structured_metrics_mode: str = "full"
    teacher_aux_mode: str = "always"

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)
    _policy_anchor_model: nn.Module | None = field(default=None, init=False)
    _amp_enabled: bool = field(default=False, init=False)
    _amp_device_type: str = field(default="cpu", init=False)
    _grad_scaler: torch.amp.GradScaler | None = field(default=None, init=False)
    _active_timing_metrics: dict[str, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.entropy_scope = str(self.entropy_scope).strip().lower()
        if self.entropy_scope not in {"candidate", "family"}:
            raise ValueError("entropy_scope must be one of: candidate, family")
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)
        self.structured_metrics_mode = str(self.structured_metrics_mode).strip().lower()
        self.teacher_aux_mode = str(self.teacher_aux_mode).strip().lower()
        self.teacher_public_heuristic_profiles = _normalize_public_heuristic_profiles(
            self.teacher_public_heuristic_profiles
        )
        self.teacher_public_heuristic_profile_mode = _normalize_public_heuristic_profile_mode(
            self.teacher_public_heuristic_profile_mode
        )
        if float(self.policy_anchor_coef) < 0.0:
            raise ValueError("policy_anchor_coef must be >= 0")
        if float(self.policy_anchor_top_action_coef) < 0.0:
            raise ValueError("policy_anchor_top_action_coef must be >= 0")
        if float(self.policy_anchor_temperature) <= 0.0:
            raise ValueError("policy_anchor_temperature must be > 0")
        if float(self.trajectory_retention_coef) < 0.0:
            raise ValueError("trajectory_retention_coef must be >= 0")
        if self.structured_metrics_mode not in {"off", "sampled", "full"}:
            raise ValueError("structured_metrics_mode must be one of: off, sampled, full")
        if self.teacher_aux_mode not in {"off", "warmstart_only", "always"}:
            raise ValueError("teacher_aux_mode must be one of: off, warmstart_only, always")
        self._refresh_acceleration_state()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def set_teacher_aux_coefs(
        self,
        *,
        family: float | None = None,
        slot: float | None = None,
        hand: float | None = None,
        move_source: float | None = None,
        attack_type: float | None = None,
        action: float | None = None,
        same_family_action: float | None = None,
        action_margin: float | None = None,
        action_margin_value: float | None = None,
        same_family_action_margin: float | None = None,
        same_family_action_margin_value: float | None = None,
        exact_action_families: tuple[str, ...] | None = None,
        public_heuristic: float | None = None,
        public_heuristic_temperature: float | None = None,
        public_nonpass_over_pass: float | None = None,
        public_nonpass_over_pass_margin: float | None = None,
        public_heuristic_families: tuple[str, ...] | None = None,
        public_heuristic_profiles: tuple[str, ...] | None = None,
        public_heuristic_profile_mode: str | None = None,
        public_heuristic_profiles_end_updates: int | None = None,
    ) -> None:
        if family is not None:
            self.teacher_family_coef = float(family)
        if slot is not None:
            self.teacher_slot_coef = float(slot)
        if hand is not None:
            self.teacher_hand_coef = float(hand)
        if move_source is not None:
            self.teacher_move_source_coef = float(move_source)
        if attack_type is not None:
            self.teacher_attack_type_coef = float(attack_type)
        if action is not None:
            self.teacher_action_coef = float(action)
        if same_family_action is not None:
            self.teacher_same_family_action_coef = float(same_family_action)
        if action_margin is not None:
            self.teacher_action_margin_coef = float(action_margin)
        if action_margin_value is not None:
            self.teacher_action_margin = float(action_margin_value)
        if same_family_action_margin is not None:
            self.teacher_same_family_action_margin_coef = float(same_family_action_margin)
        if same_family_action_margin_value is not None:
            self.teacher_same_family_action_margin = float(same_family_action_margin_value)
        if exact_action_families is not None:
            self.teacher_exact_action_families = tuple(
                str(name).strip() for name in exact_action_families if str(name).strip()
            )
        if public_heuristic is not None:
            self.teacher_public_heuristic_coef = float(public_heuristic)
        if public_heuristic_temperature is not None:
            self.teacher_public_heuristic_temperature = float(public_heuristic_temperature)
        if public_nonpass_over_pass is not None:
            self.teacher_public_nonpass_over_pass_coef = float(public_nonpass_over_pass)
        if public_nonpass_over_pass_margin is not None:
            self.teacher_public_nonpass_over_pass_margin = float(public_nonpass_over_pass_margin)
        if public_heuristic_families is not None:
            self.teacher_public_heuristic_families = tuple(
                str(name).strip() for name in public_heuristic_families if str(name).strip()
            )
        if public_heuristic_profiles is not None:
            self.teacher_public_heuristic_profiles = _normalize_public_heuristic_profiles(public_heuristic_profiles)
        if public_heuristic_profile_mode is not None:
            self.teacher_public_heuristic_profile_mode = _normalize_public_heuristic_profile_mode(
                public_heuristic_profile_mode
            )
        if public_heuristic_profiles_end_updates is not None:
            self.teacher_public_heuristic_profiles_end_updates = int(public_heuristic_profiles_end_updates)

    def _record_timing_ms(self, name: str, elapsed_seconds: float) -> None:
        record_timing_ms(
            self._active_timing_metrics,
            profile_timers=self.profile_timers,
            name=name,
            elapsed_seconds=elapsed_seconds,
        )

    def _teacher_aux_active(self, *, auxiliary_update: bool) -> bool:
        return teacher_aux_active(teacher_aux_mode=self.teacher_aux_mode, auxiliary_update=auxiliary_update)

    def _should_emit_structured_metrics(self, *, auxiliary_update: bool) -> bool:
        return should_emit_structured_metrics(
            structured_metrics_mode=self.structured_metrics_mode,
            auxiliary_update=auxiliary_update,
            update_count=self.update_count,
        )

    def _refresh_acceleration_state(self) -> None:
        self._amp_enabled, self._amp_device_type, self._grad_scaler = learner_acceleration_state(
            model=self.model,
            mixed_precision=self.mixed_precision,
        )

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        return compute_impala_loss_and_metrics_with_context(
            learner=self,
            batch=batch,
            batch_value=_batch_value,
        )
