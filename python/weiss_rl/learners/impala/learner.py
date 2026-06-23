"""IMPALA learner implementation."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from pathlib import Path as _Path
from typing import Any as _Any

import torch as _torch
from torch import Tensor as _Tensor
from torch import nn as _nn
from torch.optim import Optimizer as _Optimizer

from weiss_rl.diagnostics.training_logger import TrainingLogger as _TrainingLogger
from weiss_rl.learners.factorized_evaluation import ImpalaFactorizedEvaluationMixin as _ImpalaFactorizedEvaluationMixin
from weiss_rl.learners.impala.auxiliary_loss import ImpalaAuxiliaryLossMixin as _ImpalaAuxiliaryLossMixin
from weiss_rl.learners.impala.batch_access import batch_value as _batch_value
from weiss_rl.learners.impala.loss_pipeline import (
    compute_impala_loss_and_metrics_with_context as _compute_impala_loss_and_metrics_with_context,
)
from weiss_rl.learners.impala.policy_anchor_support import (
    ImpalaPolicyAnchorSupportMixin as _ImpalaPolicyAnchorSupportMixin,
)
from weiss_rl.learners.impala.support import ImpalaSupportMixin as _ImpalaSupportMixin
from weiss_rl.learners.impala.update_loop import ImpalaUpdateLoopMixin as _ImpalaUpdateLoopMixin
from weiss_rl.learners.structured_auxiliary import (
    normalize_public_heuristic_profile_mode as _normalize_teacher_profile_mode,
)
from weiss_rl.learners.structured_auxiliary import (
    normalize_public_heuristic_profiles as _normalize_teacher_profiles,
)
from weiss_rl.learners.update_bookkeeping import (
    learner_acceleration_state as _learner_acceleration_state,
)
from weiss_rl.learners.update_bookkeeping import (
    record_timing_ms as _record_timing_ms,
)
from weiss_rl.learners.update_bookkeeping import (
    should_emit_structured_metrics as _should_emit_structured_metrics,
)
from weiss_rl.learners.update_bookkeeping import (
    teacher_aux_active as _teacher_aux_active,
)

__all__ = ("ImpalaLearner",)


@_dataclass(slots=True)
class ImpalaLearner(
    _ImpalaUpdateLoopMixin,
    _ImpalaAuxiliaryLossMixin,
    _ImpalaFactorizedEvaluationMixin,
    _ImpalaPolicyAnchorSupportMixin,
    _ImpalaSupportMixin,
):
    """B1/Main IMPALA learner state and optimizer-facing update methods."""

    model: _nn.Module | None = None
    compiled_model: _nn.Module | None = None
    optimizer: _Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_scope: str = "candidate"
    grad_norm_clip: float = 40.0
    mixed_precision: bool = False
    checkpoint_dir: _Path | None = None
    fault_dir: _Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: _Path | None = None
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
    teacher_exact_action_families: tuple[str, ...] = _field(default_factory=tuple)
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_nonpass_over_pass_coef: float = 0.0
    teacher_public_nonpass_over_pass_margin: float = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = _field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = _field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_anchor_coef: float = 0.0
    policy_anchor_top_action_coef: float = 0.0
    policy_anchor_temperature: float = 1.0
    trajectory_retention_coef: float = 0.0
    profile_timers: bool = False
    structured_metrics_mode: str = "full"
    teacher_aux_mode: str = "always"

    update_count: int = _field(default=0, init=False)
    policy_version: int = _field(default=0, init=False)
    total_samples_processed: int = _field(default=0, init=False)
    start_time: float = _field(default_factory=_time.time, init=False)
    logger: _TrainingLogger | None = _field(default=None, init=False)
    last_log_time: float = _field(default_factory=_time.time, init=False)
    last_log_update: int = _field(default=0, init=False)
    _policy_anchor_model: _nn.Module | None = _field(default=None, init=False)
    _amp_enabled: bool = _field(default=False, init=False)
    _amp_device_type: str = _field(default="cpu", init=False)
    _grad_scaler: _torch.amp.GradScaler | None = _field(default=None, init=False)
    _active_timing_metrics: dict[str, float] | None = _field(default=None, init=False)

    def __post_init__(self) -> None:
        self._normalize_config()
        self._validate_entropy_scope()
        self._configure_logger()
        self._validate_config()
        self._refresh_acceleration_state()

    def _normalize_config(self) -> None:
        self.entropy_scope = str(self.entropy_scope).strip().lower()
        self.structured_metrics_mode = str(self.structured_metrics_mode).strip().lower()
        self.teacher_aux_mode = str(self.teacher_aux_mode).strip().lower()
        self.teacher_public_heuristic_profiles = _normalize_teacher_profiles(self.teacher_public_heuristic_profiles)
        self.teacher_public_heuristic_profile_mode = _normalize_teacher_profile_mode(
            self.teacher_public_heuristic_profile_mode
        )

    def _configure_logger(self) -> None:
        if self.logs_dir:
            self.logger = _TrainingLogger(self.logs_dir, start_time=self.start_time)

    def _validate_entropy_scope(self) -> None:
        if self.entropy_scope not in {"candidate", "family"}:
            raise ValueError("entropy_scope must be one of: candidate, family")

    def _validate_config(self) -> None:
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
            self.teacher_public_heuristic_profiles = _normalize_teacher_profiles(public_heuristic_profiles)
        if public_heuristic_profile_mode is not None:
            self.teacher_public_heuristic_profile_mode = _normalize_teacher_profile_mode(public_heuristic_profile_mode)
        if public_heuristic_profiles_end_updates is not None:
            self.teacher_public_heuristic_profiles_end_updates = int(public_heuristic_profiles_end_updates)

    def _record_timing_ms(self, name: str, elapsed_seconds: float) -> None:
        _record_timing_ms(
            self._active_timing_metrics,
            profile_timers=self.profile_timers,
            name=name,
            elapsed_seconds=elapsed_seconds,
        )

    def _teacher_aux_active(self, *, auxiliary_update: bool) -> bool:
        return _teacher_aux_active(teacher_aux_mode=self.teacher_aux_mode, auxiliary_update=auxiliary_update)

    def _should_emit_structured_metrics(self, *, auxiliary_update: bool) -> bool:
        return _should_emit_structured_metrics(
            structured_metrics_mode=self.structured_metrics_mode,
            auxiliary_update=auxiliary_update,
            update_count=self.update_count,
        )

    def _refresh_acceleration_state(self) -> None:
        self._amp_enabled, self._amp_device_type, self._grad_scaler = _learner_acceleration_state(
            model=self.model,
            mixed_precision=self.mixed_precision,
        )

    def _loss_and_metrics(self, batch: _Any) -> tuple[_Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _loss_and_metrics_with_context(self, batch: _Any) -> tuple[_Tensor, dict[str, float], dict[str, _Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        return _compute_impala_loss_and_metrics_with_context(
            learner=self,
            batch=batch,
            batch_value=_batch_value,
        )
