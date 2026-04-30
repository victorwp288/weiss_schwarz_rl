"""IMPALA learner helpers."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.learners.batch_fields import ImpalaBatchFieldsMixin as _ImpalaBatchFieldsMixin
from weiss_rl.learners.distillation_losses import ImpalaDistillationLossMixin as _ImpalaDistillationLossMixin
from weiss_rl.learners.impala_helpers import (
    _batch_value,
    _compute_vtrace_targets_torch,
    _ForwardTimeMajorResult,
    _masked_action_logp_and_entropy,
    _normalize_public_heuristic_profile_mode,
    _normalize_public_heuristic_profiles,
    _packed_action_logp_and_entropy,
    _packed_group_log_probs,
    _packed_scores_action_logp_and_entropy,
    _packed_structured_legal_view,
    _PackedStructuredLegalView,
    _resolve_public_heuristic_family_ids,
    _segment_logsumexp,
    _structured_catalog_metadata,
    _time_step_legal_actions,
    compute_structured_teacher_auxiliary_metrics,
    summarize_structured_policy_metrics,
    summarize_vtrace_diagnostics,
)
from weiss_rl.learners.impala_helpers import (
    learner_logp_from_legal_ids as learner_logp_from_legal_ids,
)
from weiss_rl.learners.impala_helpers import (
    learner_logp_from_mask as learner_logp_from_mask,
)
from weiss_rl.learners.metric_projection import build_custom_log_metrics
from weiss_rl.learners.vtrace import VtraceMetrics, VTraceTargets, compute_vtrace_metrics
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.training_logger import TrainingLogger, TrainingMetrics


@dataclass(slots=True)
class ImpalaLearner(_ImpalaBatchFieldsMixin, _ImpalaDistillationLossMixin):
    model: nn.Module | None = None
    compiled_model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_norm_clip: float = 40.0
    optimizer_backend: str = "auto"
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
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_main_move_coef: float = 0.0
    teacher_development_pass_suppression_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_loss_coef: float = 1.0
    behavior_action_bc_coef: float = 0.0
    b1_opponent_anchor_only: bool = False
    reference_policy_top_action_bc_coef: float = 0.0
    b1_opponent_reference_policy_top_action_bc_coef: float = 0.0
    b1_second_seat_positive_advantage_policy_coef: float = 0.0
    b1_second_seat_reference_top_action_avoidance_coef: float = 0.0
    reference_policy_top_action_family_bc_coef: float = 0.0
    reference_policy_model: nn.Module | None = None
    raw_b1_distill_coef: float = 0.0
    raw_b1_distill_teacher_bias_scale: float = 0.0
    raw_b1_distill_student_bias_scale: float = 0.0
    raw_b1_distill_top_k: int = 16
    raw_b1_distill_temperature: float = 1.5
    raw_b1_distill_top_action_ce_coef: float = 0.0
    counterfactual_positive_label_dirs: tuple[str, ...] = field(default_factory=tuple)
    counterfactual_positive_coef: float = 0.0
    counterfactual_positive_margin_coef: float = 0.0
    counterfactual_positive_margin: float = 1.0
    counterfactual_positive_max_labels: int = 0
    profile_timers: bool = False
    structured_metrics_mode: str = "full"
    teacher_aux_mode: str = "always"
    gradient_sync: Callable[[], None] | None = None

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)
    _amp_enabled: bool = field(default=False, init=False)
    _amp_device_type: str = field(default="cpu", init=False)
    _grad_scaler: torch.amp.GradScaler | None = field(default=None, init=False)
    _active_timing_metrics: dict[str, float] | None = field(default=None, init=False)
    _counterfactual_positive_records: tuple[dict[str, Any], ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)
        self.structured_metrics_mode = str(self.structured_metrics_mode).strip().lower()
        self.teacher_aux_mode = str(self.teacher_aux_mode).strip().lower()
        self.optimizer_backend = str(self.optimizer_backend).strip().lower()
        self.teacher_public_heuristic_profiles = _normalize_public_heuristic_profiles(
            self.teacher_public_heuristic_profiles
        )
        self.teacher_public_heuristic_profile_mode = _normalize_public_heuristic_profile_mode(
            self.teacher_public_heuristic_profile_mode
        )
        if self.structured_metrics_mode not in {"off", "sampled", "full"}:
            raise ValueError("structured_metrics_mode must be one of: off, sampled, full")
        if self.teacher_aux_mode not in {"off", "warmstart_only", "always"}:
            raise ValueError("teacher_aux_mode must be one of: off, warmstart_only, always")
        if self.optimizer_backend not in {"auto", "default", "foreach", "fused"}:
            raise ValueError("optimizer_backend must be one of: auto, default, foreach, fused")
        if int(self.counterfactual_positive_max_labels) < 0:
            raise ValueError("counterfactual_positive_max_labels must be >= 0")
        self._counterfactual_positive_records = self._load_counterfactual_positive_records(
            self.counterfactual_positive_label_dirs,
            max_labels=int(self.counterfactual_positive_max_labels),
        )
        self._refresh_acceleration_state()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def set_reference_policy_bc_coefs(
        self,
        *,
        top_action: float | None = None,
        top_action_family: float | None = None,
    ) -> None:
        if top_action is not None:
            self.reference_policy_top_action_bc_coef = float(top_action)
        if top_action_family is not None:
            self.reference_policy_top_action_family_bc_coef = float(top_action_family)

    def set_raw_b1_distill_coef(self, value: float) -> None:
        self.raw_b1_distill_coef = float(value)

    def _split_anchor_and_rl_masks(self, batch: Any, loss_mask: Tensor) -> tuple[Tensor, Tensor, dict[str, float]]:
        metrics = {
            "b1_opponent_anchor_only_active": 0.0,
            "b1_anchor_row_fraction": 0.0,
            "rl_row_fraction": 1.0,
        }
        if not bool(self.b1_opponent_anchor_only):
            return loss_mask, loss_mask, metrics

        metrics["b1_opponent_anchor_only_active"] = 1.0
        raw_b1_mask = _batch_value(batch, "b1_opponent_mask")
        if raw_b1_mask is None:
            metrics["rl_row_fraction"] = float((loss_mask > 0.0).to(dtype=loss_mask.dtype).mean().detach().item())
            return loss_mask, loss_mask.new_zeros(loss_mask.shape), metrics

        b1_mask = self._optional_time_major_loss_mask(
            raw_b1_mask,
            expected_shape=loss_mask.shape,
            like=loss_mask,
        )
        if b1_mask is None:
            metrics["rl_row_fraction"] = float((loss_mask > 0.0).to(dtype=loss_mask.dtype).mean().detach().item())
            return loss_mask, loss_mask.new_zeros(loss_mask.shape), metrics

        b1_mask = (b1_mask > 0.5).to(device=loss_mask.device, dtype=loss_mask.dtype)
        anchor_mask = loss_mask * b1_mask
        rl_mask = loss_mask * (1.0 - b1_mask)
        total_rows = torch.clamp((loss_mask > 0.0).to(dtype=loss_mask.dtype).sum(), min=1.0)
        metrics["b1_anchor_row_fraction"] = float(
            (((anchor_mask > 0.0).to(dtype=loss_mask.dtype).sum() / total_rows).detach().item())
        )
        metrics["rl_row_fraction"] = float(
            (((rl_mask > 0.0).to(dtype=loss_mask.dtype).sum() / total_rows).detach().item())
        )
        return rl_mask, anchor_mask, metrics

    def set_counterfactual_positive_coef(self, value: float) -> None:
        self.counterfactual_positive_coef = float(value)

    def set_teacher_aux_coefs(
        self,
        *,
        family: float | None = None,
        slot: float | None = None,
        move_source: float | None = None,
        attack_type: float | None = None,
        action: float | None = None,
        same_family_action: float | None = None,
        public_heuristic: float | None = None,
        public_main_move: float | None = None,
        development_pass_suppression: float | None = None,
        public_heuristic_temperature: float | None = None,
        public_heuristic_families: tuple[str, ...] | None = None,
        public_heuristic_profiles: tuple[str, ...] | None = None,
        public_heuristic_profile_mode: str | None = None,
        public_heuristic_profiles_end_updates: int | None = None,
    ) -> None:
        if family is not None:
            self.teacher_family_coef = float(family)
        if slot is not None:
            self.teacher_slot_coef = float(slot)
        if move_source is not None:
            self.teacher_move_source_coef = float(move_source)
        if attack_type is not None:
            self.teacher_attack_type_coef = float(attack_type)
        if action is not None:
            self.teacher_action_coef = float(action)
        if same_family_action is not None:
            self.teacher_same_family_action_coef = float(same_family_action)
        if public_heuristic is not None:
            self.teacher_public_heuristic_coef = float(public_heuristic)
        if public_main_move is not None:
            self.teacher_public_main_move_coef = float(public_main_move)
        if development_pass_suppression is not None:
            self.teacher_development_pass_suppression_coef = float(development_pass_suppression)
        if public_heuristic_temperature is not None:
            self.teacher_public_heuristic_temperature = float(public_heuristic_temperature)
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
        if not self.profile_timers or self._active_timing_metrics is None:
            return
        key = f"timer_{name}_ms"
        self._active_timing_metrics[key] = self._active_timing_metrics.get(key, 0.0) + (float(elapsed_seconds) * 1000.0)

    def _teacher_aux_active(self, *, auxiliary_update: bool) -> bool:
        if self.teacher_aux_mode == "off":
            return False
        if self.teacher_aux_mode == "warmstart_only":
            return bool(auxiliary_update)
        return True

    def _should_emit_structured_metrics(self, *, auxiliary_update: bool) -> bool:
        if self.structured_metrics_mode == "off":
            return False
        if self.structured_metrics_mode == "sampled":
            return (not auxiliary_update) and (int(self.update_count) % 10 == 0)
        return True

    def _refresh_acceleration_state(self) -> None:
        if self.model is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        self._amp_device_type = parameter.device.type
        self._amp_enabled = bool(self.mixed_precision and self._amp_device_type == "cuda")
        self._grad_scaler = torch.amp.GradScaler("cuda", enabled=True) if self._amp_enabled else None

    def update(self, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
        update_started = time.perf_counter()
        self.update_count += 1
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size

        elapsed = time.time() - self.start_time
        throughput_samples_per_sec = self.total_samples_processed / max(elapsed, 1e-6)
        throughput_updates_per_sec = self.update_count / max(elapsed, 1e-6)

        if self.checkpoint_dir and self.update_count % self.checkpoint_interval_updates == 0:
            self.policy_version += 1
            self._write_checkpoint_metadata()

        metrics: dict[str, float] = {
            "loss": 0.0,
            "throughput_samples_per_sec": throughput_samples_per_sec,
            "throughput_updates_per_sec": throughput_updates_per_sec,
            "entropy_coef": float(self.entropy_coef),
        }
        if self.profile_timers:
            self._active_timing_metrics = {}
        vtrace_result = _batch_value(batch, "vtrace_result")

        has_training_inputs = _batch_value(batch, "obs") is not None
        if has_training_inputs:
            missing = [key for key in ("obs", "actions") if _batch_value(batch, key) is None]
            if not self._has_legal_actions(batch):
                missing.append("legal_actions")
            has_vtrace_targets = isinstance(_batch_value(batch, "vtrace_result"), VTraceTargets)
            has_raw_vtrace_inputs = self._has_raw_vtrace_inputs(batch)
            if not has_vtrace_targets and not has_raw_vtrace_inputs:
                missing.append("vtrace_result_or_raw_inputs")
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(
                    "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs for learner updates; "
                    f"missing {missing_fields}"
                )
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model to run an optimizer step")

            self.model.train()
            if self.compiled_model is not None:
                self.compiled_model.train()
            loss_started = time.perf_counter()
            with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
                loss, loss_metrics, loss_context = self._loss_and_metrics_with_context(batch)
            self._record_timing_ms("learner_loss_and_metrics", time.perf_counter() - loss_started)
            optimizer = self._optimizer_for_step()
            backward_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_scale_before = None
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(optimizer)
            else:
                loss.backward()
            self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
            sync_started = time.perf_counter()
            self._sync_gradients_if_needed()
            self._record_timing_ms("learner_gradient_sync", time.perf_counter() - sync_started)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            optimizer_started = time.perf_counter()
            if self._grad_scaler is not None:
                grad_norm_tensor = torch.as_tensor(grad_norm)
                gradients_finite = bool(torch.isfinite(grad_norm_tensor).all().item())
                bad_gradients: dict[str, Tensor] = {}
                if not gradients_finite:
                    bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
                if gradients_finite:
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if loss_scale_before is not None:
                        try:
                            self._grad_scaler.update(loss_scale_before * 0.5)
                        except TypeError:
                            self._grad_scaler.update()
                    else:
                        self._grad_scaler.update()
                loss_scale_after = float(self._grad_scaler.get_scale())
                gradient_overflow = (not gradients_finite) or bool(
                    loss_scale_before is not None and loss_scale_after < loss_scale_before
                )
                if gradient_overflow:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 1.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
                else:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 0.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
            else:
                self._ensure_finite_gradients(batch=batch, context=loss_context, grad_norm=grad_norm)
                optimizer.step()
                metrics.update(loss_metrics)
                metrics["grad_norm"] = float(grad_norm)
            self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)

        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
            c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))

        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(metrics, batch)
            self.last_log_time = time.time()
            self.last_log_update = self.update_count

        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def auxiliary_update(self, batch: Any) -> dict[str, float]:
        """Run one optimizer step using only structured teacher supervision."""
        update_started = time.perf_counter()
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run an auxiliary optimizer step")
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size
        if self.profile_timers:
            self._active_timing_metrics = {}
        self.model.train()
        if self.compiled_model is not None:
            self.compiled_model.train()
        loss_started = time.perf_counter()
        with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
            loss, aux_metrics, aux_context = self._auxiliary_loss_and_metrics(batch)
        self._record_timing_ms("learner_auxiliary_loss_and_metrics", time.perf_counter() - loss_started)
        optimizer = self._optimizer_for_step()
        backward_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
        sync_started = time.perf_counter()
        self._sync_gradients_if_needed()
        self._record_timing_ms("learner_gradient_sync", time.perf_counter() - sync_started)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        optimizer_started = time.perf_counter()
        if self._grad_scaler is not None:
            grad_norm_tensor = torch.as_tensor(grad_norm)
            gradients_finite = bool(torch.isfinite(grad_norm_tensor).all().item())
            bad_gradients: dict[str, Tensor] = {}
            if not gradients_finite:
                bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
            if gradients_finite:
                self._grad_scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            self._grad_scaler.update()
            metrics = dict(aux_metrics)
            metrics["grad_norm"] = float(grad_norm_tensor)
            metrics["amp_grad_overflow"] = 0.0 if gradients_finite else 1.0
            metrics["loss_scale"] = float(self._grad_scaler.get_scale())
        else:
            self._ensure_finite_gradients(batch=batch, context=aux_context, grad_norm=grad_norm)
            optimizer.step()
            metrics = dict(aux_metrics)
            metrics["grad_norm"] = float(grad_norm)
        self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)
        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _evaluate_factorized_model_time_major(
        self,
        forward_model: Any,
        batch: Any,
        *,
        obs: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        actions: Tensor | None,
    ) -> Any | None:
        if not self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            return None
        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            _batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            expected_shape=expected_shape,
        )
        if acting_seat is None:
            return None
        return forward_model.evaluate_factorized_sequence_packed_seat_aware(
            obs,
            acting_seat,
            self._prepare_seat_hidden_state(
                _batch_value(batch, "initial_hidden_state"),
                batch_size=batch_size,
                like=obs,
            ),
            legal_actions=self._packed_legal_action_view(packed_legal),
            actions=actions,
        )

    def _packed_top_action_ids(
        self,
        packed_scores: Tensor,
        packed_ids: Tensor,
        packed_offsets: Tensor,
    ) -> Tensor:
        ids = packed_ids.to(device=packed_scores.device, dtype=torch.long)
        offsets = packed_offsets.to(device=packed_scores.device, dtype=torch.long)
        row_count = max(int(offsets.numel()) - 1, 0)
        if row_count == 0:
            return torch.zeros((0,), dtype=torch.long, device=packed_scores.device)
        scores = packed_scores.reshape(-1)
        widths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(row_count, device=packed_scores.device, dtype=torch.long),
            widths,
        )
        if scores.numel() == 0:
            return torch.full((row_count,), -1, dtype=torch.long, device=packed_scores.device)
        best_scores = scores.new_full((row_count,), -torch.inf)
        best_scores.scatter_reduce_(0, row_indices, scores, reduce="amax", include_self=True)
        positions = torch.arange(scores.numel(), device=packed_scores.device, dtype=torch.long)
        sentinel = torch.full_like(positions, scores.numel())
        best_positions = torch.full((row_count,), scores.numel(), device=packed_scores.device, dtype=torch.long)
        candidate_positions = torch.where(scores == best_scores.index_select(0, row_indices), positions, sentinel)
        best_positions.scatter_reduce_(0, row_indices, candidate_positions, reduce="amin", include_self=True)
        valid_rows = best_positions < scores.numel()
        safe_positions = torch.clamp(best_positions, max=max(scores.numel() - 1, 0))
        top_ids = ids.index_select(0, safe_positions)
        return torch.where(valid_rows, top_ids, torch.full_like(top_ids, -1))

    def _action_family_ids_tensor(self, *, action_dim: int, device: torch.device) -> Tensor | None:
        action_catalog = getattr(self.model, "action_catalog", None)
        if not isinstance(action_catalog, ActionCatalog):
            return None
        if int(action_catalog.action_space_size) != int(action_dim):
            return None
        family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
        family_ids = [
            int(family_index[action_catalog.decode(action_id).family])
            for action_id in range(int(action_catalog.action_space_size))
        ]
        return torch.as_tensor(family_ids, device=device, dtype=torch.long)

    def _family_log_probs_from_dense_logits(
        self,
        *,
        logits: Tensor,
        legal_mask: Tensor,
        action_family_ids: Tensor,
    ) -> Tensor:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_mask = legal_mask.to(device=logits.device, dtype=torch.bool).reshape(-1, logits.shape[-1])
        masked_logits = torch.where(flat_mask, flat_logits, torch.full_like(flat_logits, -torch.inf))
        action_log_probs = torch.log_softmax(masked_logits, dim=-1)
        family_count = int(action_family_ids.max().item()) + 1 if action_family_ids.numel() else 0
        family_log_probs = action_log_probs.new_full((flat_logits.shape[0], family_count), -torch.inf)
        for family_id in range(family_count):
            family_mask = action_family_ids == int(family_id)
            if bool(family_mask.any().item()):
                family_log_probs[:, family_id] = torch.logsumexp(action_log_probs[:, family_mask], dim=-1)
        return family_log_probs.reshape(*logits.shape[:2], family_count)

    def _reference_policy_top_action_bc_losses(
        self,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        forward_model: Any,
        packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
        legal_mask: Tensor | None,
        logits: Tensor | None,
        packed_logits: Tensor | None,
        exact_loss_enabled: bool | None = None,
        family_loss_enabled: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        zero = loss_mask.new_zeros(())
        exact_coef = float(self.reference_policy_top_action_bc_coef)
        family_coef = float(self.reference_policy_top_action_family_bc_coef)
        exact_enabled = exact_coef != 0.0 if exact_loss_enabled is None else bool(exact_loss_enabled)
        family_enabled = family_coef != 0.0 if family_loss_enabled is None else bool(family_loss_enabled)
        if self.reference_policy_model is None or (not exact_enabled and not family_enabled):
            return zero, zero
        reference_model = self.reference_policy_model
        reference_top_actions: Tensor | None = None
        current_reference_family_logp: Tensor | None = None
        if packed_legal is not None:
            with torch.no_grad():
                reference_result = self._evaluate_factorized_model_time_major(
                    reference_model,
                    batch,
                    obs=obs,
                    packed_legal=packed_legal,
                    actions=None,
                )
                if reference_result is not None:
                    reference_top_actions = getattr(reference_result, "top_action_ids", None)
                if reference_top_actions is None:
                    reference_forward = self._forward_time_major(
                        obs,
                        initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                        to_play_seat=_batch_value(batch, "to_play_seat"),
                        actor=_batch_value(batch, "actor"),
                        legal_actions=_batch_value(batch, "legal_actions"),
                        policy_train_mask=None,
                        forward_model_override=reference_model,
                    )
                    if reference_forward.packed_logits is not None:
                        reference_top_actions = self._packed_top_action_ids(
                            reference_forward.packed_logits,
                            packed_legal[0],
                            packed_legal[1],
                        ).reshape(obs.shape[:2])
                    elif reference_forward.logits is not None:
                        reference_legal_mask = self._resolve_legal_mask(
                            batch,
                            expected_shape=obs.shape[:2],
                            action_dim=reference_forward.logits.shape[-1],
                        )
                        masked_ref_logits = torch.where(
                            reference_legal_mask.to(device=reference_forward.logits.device, dtype=torch.bool),
                            reference_forward.logits,
                            torch.full_like(reference_forward.logits, -torch.inf),
                        )
                        reference_top_actions = masked_ref_logits.argmax(dim=-1)
            if reference_top_actions is None:
                return zero, zero
            reference_top_actions = reference_top_actions.to(device=obs.device, dtype=torch.long)
            safe_reference_actions = torch.clamp(reference_top_actions, min=0)
            current_reference_logp: Tensor | None = None
            current_factorized_result = self._evaluate_factorized_model_time_major(
                forward_model,
                batch,
                obs=obs,
                packed_legal=packed_legal,
                actions=safe_reference_actions,
            )
            if current_factorized_result is not None:
                current_reference_logp = getattr(current_factorized_result, "action_logp", None)
                current_family_log_probs = getattr(current_factorized_result, "family_log_probs", None)
                model_action_dim = getattr(getattr(self.model, "action_catalog", None), "action_space_size", 0)
                action_family_ids = self._action_family_ids_tensor(
                    action_dim=int(model_action_dim),
                    device=obs.device,
                )
                if current_family_log_probs is not None and action_family_ids is not None:
                    safe_family_targets = action_family_ids.index_select(
                        0, torch.clamp(safe_reference_actions.reshape(-1), max=action_family_ids.numel() - 1)
                    ).reshape_as(safe_reference_actions)
                    current_reference_family_logp = current_family_log_probs.gather(
                        -1, safe_family_targets.unsqueeze(-1)
                    ).squeeze(-1)
            elif packed_logits is not None:
                current_reference_logp, _entropy = _packed_scores_action_logp_and_entropy(
                    packed_logits,
                    packed_legal[0],
                    packed_legal[1],
                    safe_reference_actions,
                    pass_action_id=self.pass_action_id,
                )
            elif logits is not None:
                current_reference_logp, _entropy = _packed_action_logp_and_entropy(
                    logits,
                    packed_legal[0],
                    packed_legal[1],
                    safe_reference_actions,
                    pass_action_id=self.pass_action_id,
                )
            if (
                family_enabled
                and current_reference_family_logp is None
                and packed_legal[2] is not None
                and (packed_logits is not None or logits is not None)
            ):
                action_catalog = getattr(self.model, "action_catalog", None)
                action_family_ids = self._action_family_ids_tensor(
                    action_dim=int(getattr(action_catalog, "action_space_size", 0)),
                    device=obs.device,
                )
                packed_view = _packed_structured_legal_view(
                    logits=packed_logits if packed_logits is not None else logits,
                    packed_ids=packed_legal[0],
                    packed_offsets=packed_legal[1],
                    packed_meta=packed_legal[2],
                )
                if (
                    action_family_ids is not None
                    and packed_view is not None
                    and isinstance(action_catalog, ActionCatalog)
                ):
                    current_family_log_probs = _packed_group_log_probs(
                        packed_view,
                        group_ids=packed_view.family_ids,
                        group_count=len(action_catalog.families),
                    ).reshape(*obs.shape[:2], len(action_catalog.families))
                    safe_family_targets = action_family_ids.index_select(
                        0, torch.clamp(safe_reference_actions.reshape(-1), max=action_family_ids.numel() - 1)
                    ).reshape_as(safe_reference_actions)
                    current_reference_family_logp = current_family_log_probs.gather(
                        -1, safe_family_targets.unsqueeze(-1)
                    ).squeeze(-1)
            if current_reference_logp is None:
                return zero, zero
            valid_mask = (reference_top_actions >= 0).to(device=loss_mask.device, dtype=loss_mask.dtype)
        else:
            if legal_mask is None or logits is None:
                return zero, zero
            with torch.no_grad():
                flat_obs = obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2])
                ref_logits, _ref_values, _ref_hidden = reference_model(flat_obs, None)
                ref_logits = ref_logits.reshape(logits.shape)
                masked_ref_logits = torch.where(
                    legal_mask.to(device=ref_logits.device, dtype=torch.bool),
                    ref_logits,
                    torch.full_like(ref_logits, -torch.inf),
                )
                reference_top_actions = masked_ref_logits.argmax(dim=-1).to(device=obs.device, dtype=torch.long)
            current_reference_logp, _entropy = _masked_action_logp_and_entropy(
                logits,
                legal_mask,
                reference_top_actions,
                pass_action_id=self.pass_action_id,
            )
            if family_enabled:
                action_family_ids = self._action_family_ids_tensor(action_dim=logits.shape[-1], device=logits.device)
                if action_family_ids is not None:
                    current_family_log_probs = self._family_log_probs_from_dense_logits(
                        logits=logits,
                        legal_mask=legal_mask,
                        action_family_ids=action_family_ids,
                    )
                    safe_family_targets = action_family_ids.index_select(
                        0, torch.clamp(reference_top_actions.reshape(-1), max=action_family_ids.numel() - 1)
                    ).reshape_as(reference_top_actions)
                    current_reference_family_logp = current_family_log_probs.gather(
                        -1, safe_family_targets.unsqueeze(-1)
                    ).squeeze(-1)
            valid_mask = torch.isfinite(current_reference_logp).to(device=loss_mask.device, dtype=loss_mask.dtype)
        finite_reference_logp = torch.isfinite(current_reference_logp).to(
            device=loss_mask.device, dtype=loss_mask.dtype
        )
        reference_loss_mask = loss_mask * valid_mask * finite_reference_logp
        denominator = torch.clamp(reference_loss_mask.sum(), min=1.0)
        weighted_reference_logp = torch.where(
            reference_loss_mask > 0,
            current_reference_logp,
            torch.zeros_like(current_reference_logp),
        )
        exact_loss = -((weighted_reference_logp * reference_loss_mask).sum() / denominator)
        if not family_enabled or current_reference_family_logp is None:
            return exact_loss, zero
        family_valid = torch.isfinite(current_reference_family_logp).to(device=loss_mask.device, dtype=loss_mask.dtype)
        family_loss_mask = reference_loss_mask * family_valid
        family_denominator = torch.clamp(family_loss_mask.sum(), min=1.0)
        weighted_family_logp = torch.where(
            family_loss_mask > 0,
            current_reference_family_logp,
            torch.zeros_like(current_reference_family_logp),
        )
        family_loss = -((weighted_family_logp * family_loss_mask).sum() / family_denominator)
        return exact_loss, family_loss

    def _auxiliary_loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute auxiliary losses")
        action_catalog = getattr(self.model, "action_catalog", None)
        if not isinstance(action_catalog, ActionCatalog):
            raise ValueError("structured auxiliary pretraining requires a structured action catalog")
        if not self._teacher_aux_active(auxiliary_update=True):
            zero = self._model_parameter().sum() * 0.0
            return zero, {"loss": 0.0, "policy_train_fraction": 0.0}, {}

        obs = self._require_obs(_batch_value(batch, "obs"))
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        factorized_result = None
        forward_observation_context: Mapping[str, Tensor] | None = None
        if self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            factorized_result, packed_legal = self._evaluate_factorized_time_major(
                batch,
                obs=obs,
                actions=None,
            )
            logits = None
            packed_logits = None
            values = factorized_result.values
        else:
            forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=_batch_value(batch, "legal_actions"),
            )
            logits = forward.logits
            packed_logits = forward.packed_logits
            values = forward.values
            forward_observation_context = forward.observation_context
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=True)
        packed_view = None
        if packed_legal is not None and factorized_result is None:
            packed_view_started = time.perf_counter()
            packed_view = _packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
        teacher_aux_packed_view = packed_view
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        public_heuristic_target_logits = None
        if (
            packed_legal is not None
            and (float(self.teacher_public_heuristic_coef) != 0.0 or float(self.teacher_public_main_move_coef) != 0.0)
            and hasattr(forward_model, "score_packed_public_heuristic_candidates")
        ):
            public_target_rows = (
                None
                if float(self.teacher_public_main_move_coef) != 0.0
                else self._teacher_public_heuristic_rows(
                    batch,
                    loss_mask=loss_mask,
                    expected_shape=obs.shape[:2],
                    action_catalog=action_catalog,
                )
            )
            if factorized_result is not None:
                teacher_aux_packed_view, public_heuristic_target_logits = (
                    self._factorized_public_heuristic_teacher_view(
                        batch,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        active_rows=public_target_rows,
                    )
                )
            else:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    public_heuristic_target_logits = self._packed_public_heuristic_target_logits(
                        forward_model=forward_model,
                        obs=obs,
                        loss_mask=loss_mask,
                        packed_legal=packed_legal,
                        observation_context=forward_observation_context,
                        active_rows=public_target_rows,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)

        teacher_aux_started = time.perf_counter()
        teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
            logits=logits,
            legal_mask=legal_mask,
            teacher_family=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_family"),
                field_name="teacher_family",
                expected_shape=values.shape,
            ),
            teacher_slot=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_slot"),
                field_name="teacher_slot",
                expected_shape=values.shape,
            ),
            teacher_move_source=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_move_source"),
                field_name="teacher_move_source",
                expected_shape=values.shape,
            ),
            teacher_attack_type=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_attack_type"),
                field_name="teacher_attack_type",
                expected_shape=values.shape,
            ),
            teacher_action=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_action"),
                field_name="teacher_action",
                expected_shape=values.shape,
            ),
            teacher_valid=self._optional_time_major_bool_field(
                _batch_value(batch, "teacher_valid"),
                field_name="teacher_valid",
                expected_shape=values.shape,
            ),
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            family_coef=float(self.teacher_family_coef),
            slot_coef=float(self.teacher_slot_coef),
            move_source_coef=float(self.teacher_move_source_coef),
            attack_type_coef=float(self.teacher_attack_type_coef),
            action_coef=float(self.teacher_action_coef),
            same_family_action_coef=float(self.teacher_same_family_action_coef),
            public_heuristic_coef=float(self.teacher_public_heuristic_coef),
            public_main_move_coef=float(self.teacher_public_main_move_coef),
            development_pass_suppression_coef=float(self.teacher_development_pass_suppression_coef),
            public_heuristic_temperature=float(self.teacher_public_heuristic_temperature),
            public_heuristic_families=tuple(self.teacher_public_heuristic_families),
            public_heuristic_target_logits=public_heuristic_target_logits,
            packed_ids=None if packed_legal is None else packed_legal[0],
            packed_offsets=None if packed_legal is None else packed_legal[1],
            packed_meta=None if packed_legal is None else packed_legal[2],
            packed_view=teacher_aux_packed_view,
            factorized_family_log_probs=None if factorized_result is None else factorized_result.family_log_probs,
            factorized_play_slot_log_probs=None if factorized_result is None else factorized_result.play_slot_log_probs,
            factorized_move_source_log_probs=None
            if factorized_result is None
            else getattr(factorized_result, "move_source_log_probs", None),
            factorized_move_slot_log_probs=None if factorized_result is None else factorized_result.move_slot_log_probs,
            factorized_attack_slot_log_probs=None
            if factorized_result is None
            else factorized_result.attack_slot_log_probs,
            factorized_attack_type_log_probs=None
            if factorized_result is None
            else factorized_result.attack_type_log_probs,
            factorized_top_action_ids=None
            if factorized_result is None
            else getattr(factorized_result, "top_action_ids", None),
            factorized_same_family_action_logp=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_action_logp", None),
            factorized_same_family_top_action_ids=None
            if factorized_result is None
            else getattr(factorized_result, "same_family_top_action_ids", None),
        )
        self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
        context: dict[str, Any] = {
            "auxiliary_loss": teacher_aux_loss.detach(),
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
            "policy_train_mask": loss_mask.detach(),
            **teacher_context,
        }
        if factorized_result is not None:
            context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
        self._ensure_finite_tensor("auxiliary_loss", teacher_aux_loss, batch=batch, context=context)
        metrics = {
            "loss": float(teacher_aux_loss.detach().item()),
            "policy_train_fraction": float(loss_mask.mean().detach().item()),
        }
        metrics.update(teacher_metrics)
        if emit_structured_metrics:
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                    factorized_family_log_probs=None
                    if factorized_result is None
                    else factorized_result.family_log_probs,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return teacher_aux_loss, metrics, context

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        vtrace_result = _batch_value(batch, "vtrace_result")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=obs.shape[:2],
            like=obs[..., 0],
        )
        if loss_mask is None:
            loss_mask = torch.ones(obs.shape[:2], dtype=obs.dtype, device=obs.device)
        rl_loss_mask, b1_anchor_mask, anchor_mask_metrics = self._split_anchor_and_rl_masks(batch, loss_mask)
        action_catalog = getattr(self.model, "action_catalog", None)
        teacher_aux_active = isinstance(action_catalog, ActionCatalog) and self._teacher_aux_active(
            auxiliary_update=False
        )
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=False)
        restrict_packed_policy_rows = bool(
            packed_legal is not None and bool((loss_mask <= 0.0).any().item()) and not emit_structured_metrics
        )
        factorized_result = None
        forward_observation_context: Mapping[str, Tensor] | None = None
        if self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            factorized_result, packed_legal = self._evaluate_factorized_time_major(
                batch,
                obs=obs,
                actions=actions,
            )
            logits = None
            packed_logits = None
            values = factorized_result.values
        else:
            forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=_batch_value(batch, "legal_actions"),
                policy_train_mask=loss_mask if restrict_packed_policy_rows else None,
            )
            logits = forward.logits
            packed_logits = forward.packed_logits
            values = forward.values
            forward_observation_context = forward.observation_context
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")
        packed_view = None
        if packed_legal is not None and factorized_result is None and (emit_structured_metrics or teacher_aux_active):
            packed_view_started = time.perf_counter()
            packed_view = _packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
        teacher_aux_packed_view = packed_view
        public_heuristic_target_logits = None
        if (
            teacher_aux_active
            and packed_legal is not None
            and (float(self.teacher_public_heuristic_coef) != 0.0 or float(self.teacher_public_main_move_coef) != 0.0)
            and hasattr(forward_model, "score_packed_public_heuristic_candidates")
        ):
            assert isinstance(action_catalog, ActionCatalog)
            public_target_rows = (
                None
                if float(self.teacher_public_main_move_coef) != 0.0
                else self._teacher_public_heuristic_rows(
                    batch,
                    loss_mask=rl_loss_mask,
                    expected_shape=obs.shape[:2],
                    action_catalog=action_catalog,
                )
            )
            if factorized_result is not None:
                teacher_aux_packed_view, public_heuristic_target_logits = (
                    self._factorized_public_heuristic_teacher_view(
                        batch,
                        obs=obs,
                        loss_mask=rl_loss_mask,
                        packed_legal=packed_legal,
                        active_rows=public_target_rows,
                    )
                )
            else:
                heuristic_started = time.perf_counter()
                with torch.no_grad():
                    public_heuristic_target_logits = self._packed_public_heuristic_target_logits(
                        forward_model=forward_model,
                        obs=obs,
                        loss_mask=rl_loss_mask,
                        packed_legal=packed_legal,
                        observation_context=forward_observation_context,
                        active_rows=public_target_rows,
                    )
                self._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)

        context: dict[str, Any] = {
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
        }
        if logits is not None:
            self._ensure_finite_tensor("forward_logits", logits, batch=batch, context=context)
        if packed_logits is not None:
            self._ensure_finite_tensor("forward_packed_logits", packed_logits, batch=batch, context=context)
        self._ensure_finite_tensor("forward_values", values, batch=batch, context=context)

        if factorized_result is not None:
            if factorized_result.action_logp is None or factorized_result.entropy is None:
                raise ValueError("factorized learner path requires action_logp and entropy")
            action_logp = factorized_result.action_logp
            entropy = factorized_result.entropy
        elif packed_legal is not None:
            packed_reductions_started = time.perf_counter()
            packed_ids, packed_offsets, _packed_meta = packed_legal
            if packed_logits is not None:
                action_logp, entropy = _packed_scores_action_logp_and_entropy(
                    packed_logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
            else:
                assert logits is not None
                action_logp, entropy = _packed_action_logp_and_entropy(
                    logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
            self._record_timing_ms("learner_packed_reductions", time.perf_counter() - packed_reductions_started)
        else:
            assert legal_mask is not None
            assert logits is not None
            action_logp, entropy = _masked_action_logp_and_entropy(
                logits,
                legal_mask,
                actions,
                pass_action_id=self.pass_action_id,
            )
        context["action_logp"] = action_logp.detach()
        context["entropy"] = entropy.detach()
        self._ensure_finite_tensor("action_logp", action_logp, batch=batch, context=context)
        self._ensure_finite_tensor("entropy", entropy, batch=batch, context=context)

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
        c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
        raw_behavior_logp = _batch_value(batch, "behavior_logp")
        behavior_logp_for_mask = None
        if raw_behavior_logp is not None:
            behavior_logp_for_mask = self._float_target(
                raw_behavior_logp,
                expected_shape=values.shape,
                like=values,
            )
        restrict_vtrace_to_behavior = bool(
            behavior_logp_for_mask is not None and bool((loss_mask <= 0.0).any().item()) and restrict_packed_policy_rows
        )
        if restrict_vtrace_to_behavior:
            action_logp = torch.where(loss_mask > 0.0, action_logp, behavior_logp_for_mask)
        if isinstance(vtrace_result, VTraceTargets):
            targets = self._float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
            advantages = self._float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)
            rhos_for_metrics = self._float_target(vtrace_result.rhos, expected_shape=values.shape, like=values)
            raw_rewards = _batch_value(batch, "rewards")
            if raw_rewards is None:
                rewards_for_metrics = torch.zeros_like(values)
            else:
                rewards_for_metrics = self._float_target(raw_rewards, expected_shape=values.shape, like=values)
        else:
            rewards = self._float_target(_batch_value(batch, "rewards"), expected_shape=values.shape, like=values)
            discounts = self._float_target(_batch_value(batch, "discounts"), expected_shape=values.shape, like=values)
            if behavior_logp_for_mask is None:
                raise ValueError("raw V-trace batches must include behavior_logp")
            behavior_logp = behavior_logp_for_mask
            bootstrap_value = self._resolve_vtrace_bootstrap_value(
                batch,
                batch_size=int(values.shape[1]),
                like=values,
            )
            full_values = torch.cat([values.detach(), bootstrap_value.detach().unsqueeze(0)], dim=0)
            # Use the current learner policy log-prob for the V-trace target policy.
            # Passing behavior_logp twice silently forces rho=1 and disables off-policy correction.
            targets, advantages, rhos_for_metrics = _compute_vtrace_targets_torch(
                rewards,
                full_values,
                discounts,
                behavior_logp,
                action_logp,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
            rewards_for_metrics = rewards
        context["targets"] = targets.detach()
        context["advantages"] = advantages.detach()
        context["vtrace_rhos"] = rhos_for_metrics.detach()
        context["rewards"] = rewards_for_metrics.detach()
        context["policy_train_mask"] = loss_mask.detach()
        context["rl_train_mask"] = rl_loss_mask.detach()
        context["b1_anchor_mask"] = b1_anchor_mask.detach()
        loss_denominator = torch.clamp(rl_loss_mask.sum(), min=1.0)

        policy_loss = -((action_logp * advantages) * rl_loss_mask).sum() / loss_denominator
        value_loss = (((values - targets) ** 2) * rl_loss_mask).sum() / loss_denominator
        entropy_mean = (entropy * rl_loss_mask).sum() / loss_denominator
        behavior_action_bc_loss = -(action_logp * rl_loss_mask).sum() / loss_denominator
        reference_policy_top_action_bc_loss, reference_policy_top_action_family_bc_loss = (
            self._reference_policy_top_action_bc_losses(
                batch,
                obs=obs,
                loss_mask=b1_anchor_mask,
                forward_model=forward_model,
                packed_legal=packed_legal,
                legal_mask=legal_mask,
                logits=logits,
                packed_logits=packed_logits,
            )
        )
        raw_b1_distill_loss, raw_b1_distill_metrics = self._raw_b1_distill_loss_and_metrics(
            batch,
            obs=obs,
            loss_mask=b1_anchor_mask,
            packed_legal=packed_legal,
            legal_mask=legal_mask,
        )
        counterfactual_positive_loss, counterfactual_positive_metrics = self._counterfactual_positive_loss_and_metrics()
        b1_reference_policy_top_action_bc_loss = loss_mask.new_zeros(())
        b1_reference_policy_top_action_bc_row_fraction = loss_mask.new_zeros(())
        raw_b1_opponent_mask = _batch_value(batch, "b1_opponent_mask")
        b1_opponent_mask = self._optional_time_major_loss_mask(
            raw_b1_opponent_mask,
            expected_shape=obs.shape[:2],
            like=loss_mask,
        )
        if float(self.b1_opponent_reference_policy_top_action_bc_coef) != 0.0:
            if b1_opponent_mask is not None:
                b1_loss_mask = loss_mask * b1_opponent_mask
                b1_reference_policy_top_action_bc_row_fraction = b1_loss_mask.sum() / torch.clamp(
                    loss_mask.sum(),
                    min=1.0,
                )
                if bool((b1_loss_mask > 0.0).any().item()):
                    b1_reference_policy_top_action_bc_loss, _b1_family_loss = (
                        self._reference_policy_top_action_bc_losses(
                            batch,
                            obs=obs,
                            loss_mask=b1_loss_mask,
                            forward_model=forward_model,
                            packed_legal=packed_legal,
                            legal_mask=legal_mask,
                            logits=logits,
                            packed_logits=packed_logits,
                            exact_loss_enabled=True,
                            family_loss_enabled=False,
                        )
                    )
        b1_second_seat_positive_advantage_policy_loss = loss_mask.new_zeros(())
        b1_second_seat_positive_advantage_row_fraction = loss_mask.new_zeros(())
        b1_second_seat_positive_advantage_mean = loss_mask.new_zeros(())
        if float(self.b1_second_seat_positive_advantage_policy_coef) != 0.0 and b1_opponent_mask is not None:
            raw_to_play_seat = _batch_value(batch, "to_play_seat")
            if raw_to_play_seat is None:
                raw_to_play_seat = _batch_value(batch, "actor")
            if raw_to_play_seat is not None:
                acting_seat = self._optional_time_major_index_field(
                    raw_to_play_seat,
                    field_name="to_play_seat",
                    expected_shape=obs.shape[:2],
                )
                assert acting_seat is not None
                positive_advantages = torch.clamp(advantages.detach(), min=0.0)
                b1_second_seat_mask = (
                    loss_mask
                    * b1_opponent_mask
                    * (acting_seat == 1).to(device=loss_mask.device, dtype=loss_mask.dtype)
                    * (positive_advantages > 0.0).to(device=loss_mask.device, dtype=loss_mask.dtype)
                )
                b1_second_seat_positive_advantage_row_fraction = b1_second_seat_mask.sum() / torch.clamp(
                    loss_mask.sum(),
                    min=1.0,
                )
                b1_second_seat_denominator = torch.clamp(b1_second_seat_mask.sum(), min=1.0)
                weighted_positive_advantages = torch.where(
                    b1_second_seat_mask > 0.0,
                    positive_advantages,
                    torch.zeros_like(positive_advantages),
                )
                b1_second_seat_positive_advantage_mean = weighted_positive_advantages.sum() / b1_second_seat_denominator
                b1_second_seat_positive_advantage_policy_loss = -(
                    (action_logp * weighted_positive_advantages * b1_second_seat_mask).sum()
                    / b1_second_seat_denominator
                )
        b1_second_seat_reference_top_action_avoidance_loss = loss_mask.new_zeros(())
        b1_second_seat_reference_top_action_avoidance_row_fraction = loss_mask.new_zeros(())
        if float(self.b1_second_seat_reference_top_action_avoidance_coef) != 0.0 and b1_opponent_mask is not None:
            raw_to_play_seat = _batch_value(batch, "to_play_seat")
            if raw_to_play_seat is None:
                raw_to_play_seat = _batch_value(batch, "actor")
            if raw_to_play_seat is not None:
                acting_seat = self._optional_time_major_index_field(
                    raw_to_play_seat,
                    field_name="to_play_seat",
                    expected_shape=obs.shape[:2],
                )
                assert acting_seat is not None
                b1_second_seat_avoidance_mask = (
                    loss_mask * b1_opponent_mask * (acting_seat == 1).to(device=loss_mask.device, dtype=loss_mask.dtype)
                )
                b1_second_seat_reference_top_action_avoidance_row_fraction = (
                    b1_second_seat_avoidance_mask.sum() / torch.clamp(loss_mask.sum(), min=1.0)
                )
                if bool((b1_second_seat_avoidance_mask > 0.0).any().item()):
                    b1_second_seat_reference_top_action_avoidance_loss, _b1_avoid_family_loss = (
                        self._reference_policy_top_action_bc_losses(
                            batch,
                            obs=obs,
                            loss_mask=b1_second_seat_avoidance_mask,
                            forward_model=forward_model,
                            packed_legal=packed_legal,
                            legal_mask=legal_mask,
                            logits=logits,
                            packed_logits=packed_logits,
                            exact_loss_enabled=True,
                            family_loss_enabled=False,
                        )
                    )
        total_loss = (
            (float(self.policy_loss_coef) * policy_loss)
            + (self.value_loss_coef * value_loss)
            - (self.entropy_coef * entropy_mean)
        )
        if float(self.behavior_action_bc_coef) != 0.0:
            total_loss = total_loss + (float(self.behavior_action_bc_coef) * behavior_action_bc_loss)
        if float(self.reference_policy_top_action_bc_coef) != 0.0:
            total_loss = total_loss + (
                float(self.reference_policy_top_action_bc_coef) * reference_policy_top_action_bc_loss
            )
        if float(self.reference_policy_top_action_family_bc_coef) != 0.0:
            total_loss = total_loss + (
                float(self.reference_policy_top_action_family_bc_coef) * reference_policy_top_action_family_bc_loss
            )
        if float(self.b1_opponent_reference_policy_top_action_bc_coef) != 0.0:
            total_loss = total_loss + (
                float(self.b1_opponent_reference_policy_top_action_bc_coef) * b1_reference_policy_top_action_bc_loss
            )
        if float(self.b1_second_seat_positive_advantage_policy_coef) != 0.0:
            total_loss = total_loss + (
                float(self.b1_second_seat_positive_advantage_policy_coef)
                * b1_second_seat_positive_advantage_policy_loss
            )
        if float(self.b1_second_seat_reference_top_action_avoidance_coef) != 0.0:
            total_loss = total_loss - (
                float(self.b1_second_seat_reference_top_action_avoidance_coef)
                * b1_second_seat_reference_top_action_avoidance_loss
            )
        if float(self.raw_b1_distill_coef) != 0.0:
            total_loss = total_loss + (float(self.raw_b1_distill_coef) * raw_b1_distill_loss)
        if float(self.counterfactual_positive_coef) != 0.0:
            total_loss = total_loss + (float(self.counterfactual_positive_coef) * counterfactual_positive_loss)

        teacher_metrics: dict[str, float] = {}
        if teacher_aux_active:
            structured_legal_mask = (
                None
                if factorized_result is not None
                else (
                    legal_mask
                    if legal_mask is not None
                    else (
                        None
                        if packed_legal is not None and packed_legal[2] is not None
                        else self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
                    )
                )
            )
            teacher_aux_started = time.perf_counter()
            teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
                logits=logits,
                legal_mask=structured_legal_mask,
                teacher_family=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_family"),
                    field_name="teacher_family",
                    expected_shape=values.shape,
                ),
                teacher_slot=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_slot"),
                    field_name="teacher_slot",
                    expected_shape=values.shape,
                ),
                teacher_move_source=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_move_source"),
                    field_name="teacher_move_source",
                    expected_shape=values.shape,
                ),
                teacher_attack_type=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_attack_type"),
                    field_name="teacher_attack_type",
                    expected_shape=values.shape,
                ),
                teacher_action=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_action"),
                    field_name="teacher_action",
                    expected_shape=values.shape,
                ),
                teacher_valid=self._optional_time_major_bool_field(
                    _batch_value(batch, "teacher_valid"),
                    field_name="teacher_valid",
                    expected_shape=values.shape,
                ),
                loss_mask=rl_loss_mask,
                action_catalog=action_catalog,
                family_coef=float(self.teacher_family_coef),
                slot_coef=float(self.teacher_slot_coef),
                move_source_coef=float(self.teacher_move_source_coef),
                attack_type_coef=float(self.teacher_attack_type_coef),
                action_coef=float(self.teacher_action_coef),
                same_family_action_coef=float(self.teacher_same_family_action_coef),
                public_heuristic_coef=float(self.teacher_public_heuristic_coef),
                public_main_move_coef=float(self.teacher_public_main_move_coef),
                development_pass_suppression_coef=float(self.teacher_development_pass_suppression_coef),
                public_heuristic_temperature=float(self.teacher_public_heuristic_temperature),
                public_heuristic_families=tuple(self.teacher_public_heuristic_families),
                public_heuristic_target_logits=public_heuristic_target_logits,
                packed_ids=None if packed_legal is None else packed_legal[0],
                packed_offsets=None if packed_legal is None else packed_legal[1],
                packed_meta=None if packed_legal is None else packed_legal[2],
                packed_view=teacher_aux_packed_view,
                factorized_family_log_probs=None if factorized_result is None else factorized_result.family_log_probs,
                factorized_play_slot_log_probs=None
                if factorized_result is None
                else factorized_result.play_slot_log_probs,
                factorized_move_source_log_probs=None
                if factorized_result is None
                else getattr(factorized_result, "move_source_log_probs", None),
                factorized_move_slot_log_probs=None
                if factorized_result is None
                else factorized_result.move_slot_log_probs,
                factorized_attack_slot_log_probs=None
                if factorized_result is None
                else factorized_result.attack_slot_log_probs,
                factorized_attack_type_log_probs=None
                if factorized_result is None
                else factorized_result.attack_type_log_probs,
                factorized_top_action_ids=None
                if factorized_result is None
                else getattr(factorized_result, "top_action_ids", None),
                factorized_same_family_action_logp=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_action_logp", None),
                factorized_same_family_top_action_ids=None
                if factorized_result is None
                else getattr(factorized_result, "same_family_top_action_ids", None),
            )
            self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
            total_loss = total_loss + teacher_aux_loss
            context.update(teacher_context)

        context["policy_loss"] = policy_loss.detach()
        context["value_loss"] = value_loss.detach()
        context["entropy_mean"] = entropy_mean.detach()
        context["behavior_action_bc_loss"] = behavior_action_bc_loss.detach()
        context["reference_policy_top_action_bc_loss"] = reference_policy_top_action_bc_loss.detach()
        context["reference_policy_top_action_family_bc_loss"] = reference_policy_top_action_family_bc_loss.detach()
        context["raw_b1_distill_loss"] = raw_b1_distill_loss.detach()
        context["counterfactual_positive_loss"] = counterfactual_positive_loss.detach()
        context["b1_opponent_reference_policy_top_action_bc_loss"] = b1_reference_policy_top_action_bc_loss.detach()
        context["b1_second_seat_positive_advantage_policy_loss"] = (
            b1_second_seat_positive_advantage_policy_loss.detach()
        )
        context["b1_second_seat_reference_top_action_avoidance_loss"] = (
            b1_second_seat_reference_top_action_avoidance_loss.detach()
        )
        context["total_loss"] = total_loss.detach()
        if factorized_result is not None:
            context["factorized_family_log_probs"] = factorized_result.family_log_probs.detach()
        self._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
        self._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
        self._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
        self._ensure_finite_tensor("behavior_action_bc_loss", behavior_action_bc_loss, batch=batch, context=context)
        self._ensure_finite_tensor(
            "reference_policy_top_action_bc_loss",
            reference_policy_top_action_bc_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor(
            "reference_policy_top_action_family_bc_loss",
            reference_policy_top_action_family_bc_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor("raw_b1_distill_loss", raw_b1_distill_loss, batch=batch, context=context)
        self._ensure_finite_tensor(
            "counterfactual_positive_loss",
            counterfactual_positive_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor(
            "b1_opponent_reference_policy_top_action_bc_loss",
            b1_reference_policy_top_action_bc_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor(
            "b1_second_seat_positive_advantage_policy_loss",
            b1_second_seat_positive_advantage_policy_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor(
            "b1_second_seat_reference_top_action_avoidance_loss",
            b1_second_seat_reference_top_action_avoidance_loss,
            batch=batch,
            context=context,
        )
        self._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)

        rho_metrics = rhos_for_metrics.detach().reshape(-1).to(dtype=torch.float32)
        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "policy_loss_coef": float(self.policy_loss_coef),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "behavior_action_bc_loss": float(behavior_action_bc_loss.detach()),
            "behavior_action_bc_coef": float(self.behavior_action_bc_coef),
            "reference_policy_top_action_bc_loss": float(reference_policy_top_action_bc_loss.detach()),
            "reference_policy_top_action_bc_coef": float(self.reference_policy_top_action_bc_coef),
            "reference_policy_top_action_family_bc_loss": float(reference_policy_top_action_family_bc_loss.detach()),
            "reference_policy_top_action_family_bc_coef": float(self.reference_policy_top_action_family_bc_coef),
            "raw_b1_distill_loss": float(raw_b1_distill_loss.detach()),
            "raw_b1_distill_coef": float(self.raw_b1_distill_coef),
            "raw_b1_distill_teacher_bias_scale": float(self.raw_b1_distill_teacher_bias_scale),
            "raw_b1_distill_student_bias_scale": float(self.raw_b1_distill_student_bias_scale),
            "raw_b1_distill_temperature": float(self.raw_b1_distill_temperature),
            "raw_b1_distill_top_k": float(self.raw_b1_distill_top_k),
            "raw_b1_distill_top_action_ce_coef": float(self.raw_b1_distill_top_action_ce_coef),
            "raw_b1_distill_row_fraction": float(raw_b1_distill_metrics["raw_b1_distill_row_fraction"].detach()),
            "raw_b1_top1_match": float(raw_b1_distill_metrics["raw_b1_top1_match"].detach()),
            "raw_b1_topk_overlap": float(raw_b1_distill_metrics["raw_b1_topk_overlap"].detach()),
            "raw_b1_family_match": float(raw_b1_distill_metrics["raw_b1_family_match"].detach()),
            "raw_b1_kl": float(raw_b1_distill_metrics["raw_b1_kl"].detach()),
            "raw_b1_top_action_ce": float(raw_b1_distill_metrics["raw_b1_top_action_ce"].detach()),
            "counterfactual_positive_loss": float(counterfactual_positive_loss.detach()),
            "counterfactual_positive_coef": float(self.counterfactual_positive_coef),
            "counterfactual_positive_margin_coef": float(self.counterfactual_positive_margin_coef),
            "counterfactual_positive_margin": float(self.counterfactual_positive_margin),
            "counterfactual_positive_ce_loss": float(
                counterfactual_positive_metrics["counterfactual_positive_ce_loss"].detach()
            ),
            "counterfactual_positive_margin_loss": float(
                counterfactual_positive_metrics["counterfactual_positive_margin_loss"].detach()
            ),
            "counterfactual_positive_label_count": float(
                counterfactual_positive_metrics["counterfactual_positive_label_count"].detach()
            ),
            "counterfactual_positive_weight_mean": float(
                counterfactual_positive_metrics["counterfactual_positive_weight_mean"].detach()
            ),
            "counterfactual_positive_prob_mean": float(
                counterfactual_positive_metrics["counterfactual_positive_prob_mean"].detach()
            ),
            "counterfactual_positive_top1_match": float(
                counterfactual_positive_metrics["counterfactual_positive_top1_match"].detach()
            ),
            "counterfactual_positive_logit_margin_mean": float(
                counterfactual_positive_metrics["counterfactual_positive_logit_margin_mean"].detach()
            ),
            "b1_opponent_reference_policy_top_action_bc_loss": float(b1_reference_policy_top_action_bc_loss.detach()),
            "b1_opponent_reference_policy_top_action_bc_coef": float(
                self.b1_opponent_reference_policy_top_action_bc_coef
            ),
            "b1_opponent_reference_policy_top_action_bc_row_fraction": float(
                b1_reference_policy_top_action_bc_row_fraction.detach()
            ),
            "b1_second_seat_positive_advantage_policy_loss": float(
                b1_second_seat_positive_advantage_policy_loss.detach()
            ),
            "b1_second_seat_positive_advantage_policy_coef": float(self.b1_second_seat_positive_advantage_policy_coef),
            "b1_second_seat_positive_advantage_row_fraction": float(
                b1_second_seat_positive_advantage_row_fraction.detach()
            ),
            "b1_second_seat_positive_advantage_mean": float(b1_second_seat_positive_advantage_mean.detach()),
            "b1_second_seat_reference_top_action_avoidance_loss": float(
                b1_second_seat_reference_top_action_avoidance_loss.detach()
            ),
            "b1_second_seat_reference_top_action_avoidance_coef": float(
                self.b1_second_seat_reference_top_action_avoidance_coef
            ),
            "b1_second_seat_reference_top_action_avoidance_row_fraction": float(
                b1_second_seat_reference_top_action_avoidance_row_fraction.detach()
            ),
            "policy_train_fraction": float(loss_mask.mean().detach()),
            "reward_mean": float(rewards_for_metrics.detach().mean().item()),
            "reward_abs_mean": float(rewards_for_metrics.detach().abs().mean().item()),
            "reward_nonzero_fraction": float((rewards_for_metrics.detach() != 0).float().mean().item()),
            "advantage_mean": float(advantages.detach().mean().item()),
            "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
            "target_mean": float(targets.detach().mean().item()),
            "target_abs_mean": float(targets.detach().abs().mean().item()),
            "vtrace_rho_p50": float(torch.quantile(rho_metrics, 0.50).item()),
            "vtrace_rho_p90": float(torch.quantile(rho_metrics, 0.90).item()),
            "vtrace_rho_p95": float(torch.quantile(rho_metrics, 0.95).item()),
            "vtrace_rho_p99": float(torch.quantile(rho_metrics, 0.99).item()),
            "vtrace_rho_clip_rate": float((rhos_for_metrics.detach() > rho_bar).float().mean().item()),
            "vtrace_c_clip_rate": float((rhos_for_metrics.detach() > c_bar).float().mean().item()),
        }
        metrics.update(anchor_mask_metrics)
        metrics.update(teacher_metrics)
        if isinstance(action_catalog, ActionCatalog) and emit_structured_metrics:
            structured_legal_mask = (
                None
                if factorized_result is not None
                else (
                    legal_mask
                    if legal_mask is not None
                    else (
                        None
                        if packed_legal is not None and packed_legal[2] is not None
                        else self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
                    )
                )
            )
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    structured_legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                    factorized_family_log_probs=None
                    if factorized_result is None
                    else factorized_result.family_log_probs,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return total_loss, metrics, context

    def _optimizer_for_step(self) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = self._create_adam_optimizer()
        return self.optimizer

    def _sync_gradients_if_needed(self) -> None:
        if self.gradient_sync is None:
            return
        self.gradient_sync()

    def _create_adam_optimizer(self) -> Optimizer:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model before creating an optimizer")
        params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("ImpalaLearner requires trainable parameters before creating an optimizer")
        kwargs: dict[str, bool] = {}
        backend = str(self.optimizer_backend).strip().lower()
        device_type = params[0].device.type
        if backend == "auto":
            if device_type == "cuda":
                kwargs["fused"] = True
        elif backend == "foreach":
            kwargs["foreach"] = True
        elif backend == "fused":
            kwargs["fused"] = True
        try:
            return torch.optim.Adam(params, lr=self.learning_rate, **kwargs)
        except (RuntimeError, TypeError):
            if backend in {"auto", "foreach", "fused"}:
                return torch.optim.Adam(params, lr=self.learning_rate)
            raise

    def _forward_time_major(
        self,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: LegalActionBatch | None = None,
        policy_train_mask: Tensor | None = None,
        forward_model_override: Any | None = None,
    ) -> _ForwardTimeMajorResult:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        forward_model = (
            forward_model_override
            if forward_model_override is not None
            else (self.compiled_model if self.compiled_model is not None else self.model)
        )
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        structured_legal_actions = bool(getattr(forward_model, "supports_legal_candidate_scoring", False))
        acting_seat = self._prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
        )
        if structured_legal_actions and legal_actions is not None:
            if legal_actions.ids is None or legal_actions.offsets is None:
                raise ValueError("structured learner updates require packed legal_actions ids/offsets")
            if legal_actions.meta is None:
                raise ValueError("structured learner updates require packed legal_actions metadata")
        sequence_started = time.perf_counter()
        if (
            acting_seat is not None
            and structured_legal_actions
            and legal_actions is not None
            and hasattr(forward_model, "forward_trunk_sequence_seat_aware")
        ):
            trunk_started = time.perf_counter()
            recurrent_flat, state_repr, observation_context, values, _next_hidden = (
                forward_model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
                )
            )
            self._record_timing_ms("learner_trunk", time.perf_counter() - trunk_started)
            restricted_rows = (
                policy_train_mask.reshape(-1).to(device=recurrent_flat.device, dtype=torch.bool)
                if policy_train_mask is not None
                else None
            )
            active_rows = None if restricted_rows is None else torch.nonzero(restricted_rows, as_tuple=False).squeeze(1)
            packed_logits: Tensor
            scorer_started = time.perf_counter()
            if active_rows is None or int(active_rows.shape[0]) == int(recurrent_flat.shape[0]):
                packed_logits = forward_model.score_packed_legal_candidates(
                    recurrent_flat,
                    obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
                    legal_actions,
                    state_repr=state_repr,
                    observation_context=observation_context,
                    scoring_mode="learner",
                )
            else:
                packed_legal = (
                    torch.as_tensor(legal_actions.ids, device=recurrent_flat.device, dtype=torch.long),
                    torch.as_tensor(legal_actions.offsets, device=recurrent_flat.device, dtype=torch.long),
                    torch.as_tensor(legal_actions.meta, device=recurrent_flat.device, dtype=torch.long),
                )
                subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
                subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
                subset_logits = (
                    recurrent_flat.new_zeros((0,))
                    if active_rows.numel() == 0
                    else torch.as_tensor(
                        forward_model.score_packed_legal_candidates(
                            recurrent_flat.index_select(0, active_rows),
                            obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]).index_select(0, active_rows),
                            subset_legal_actions,
                            state_repr=state_repr.index_select(0, active_rows),
                            observation_context=self._subset_observation_context_rows(
                                observation_context,
                                active_rows,
                                row_count=int(recurrent_flat.shape[0]),
                            ),
                            scoring_mode="learner",
                        ),
                        device=recurrent_flat.device,
                    )
                )
                packed_logits = self._scatter_packed_candidate_values(
                    packed_legal,
                    active_rows,
                    subset_logits,
                    fill_value=0.0,
                )
            self._record_timing_ms("learner_packed_scorer", time.perf_counter() - scorer_started)
            self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
            packed_rows = int(legal_actions.offsets.shape[0] - 1)
            packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
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
            if self._active_timing_metrics is not None:
                self._active_timing_metrics.update(metrics)
            return _ForwardTimeMajorResult(
                packed_logits=torch.as_tensor(packed_logits),
                values=torch.as_tensor(values),
                observation_context=observation_context,
            )
        if (
            acting_seat is not None
            and structured_legal_actions
            and legal_actions is not None
            and hasattr(forward_model, "forward_sequence_packed_seat_aware")
        ):
            packed_logits, values, _next_hidden = forward_model.forward_sequence_packed_seat_aware(
                obs,
                acting_seat,
                self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
                legal_actions=legal_actions,
                scoring_mode="learner",
            )
            self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
            packed_rows = int(legal_actions.offsets.shape[0] - 1)
            packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
            if self._active_timing_metrics is not None:
                self._active_timing_metrics.update(
                    {
                        "packed_candidate_count": float(packed_candidates),
                        "packed_candidate_rows": float(packed_rows),
                        "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
                    }
                )
            return _ForwardTimeMajorResult(
                packed_logits=torch.as_tensor(packed_logits),
                values=torch.as_tensor(values),
            )
        if acting_seat is not None and hasattr(forward_model, "forward_sequence_seat_aware"):
            logits, values, _next_hidden = forward_model.forward_sequence_seat_aware(
                obs,
                acting_seat,
                self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
                legal_actions=legal_actions if structured_legal_actions else None,
            )
            self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
            metrics = {}
            if structured_legal_actions and legal_actions is not None and legal_actions.offsets is not None:
                packed_rows = int(legal_actions.offsets.shape[0] - 1)
                packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
                metrics = {
                    "packed_candidate_count": float(packed_candidates),
                    "packed_candidate_rows": float(packed_rows),
                    "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
                }
            if self._active_timing_metrics is not None:
                self._active_timing_metrics.update(metrics)
            return _ForwardTimeMajorResult(
                logits=torch.as_tensor(logits),
                values=torch.as_tensor(values),
            )
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []

        if acting_seat is None:
            hidden_state = self._prepare_legacy_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
            for step_index, step_obs in enumerate(obs.unbind(dim=0)):
                step_legal_actions = (
                    _time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
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
                hidden_state = torch.as_tensor(hidden_state)
            return _ForwardTimeMajorResult(
                logits=torch.stack(logits_steps, dim=0),
                values=torch.stack(value_steps, dim=0),
            )

        seat_hidden_state = self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
        for step_index, (step_obs, step_seat) in enumerate(
            zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)
        ):
            step_legal_actions = (
                _time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
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
            seat_hidden_state = torch.as_tensor(seat_hidden_state)
        self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
        return _ForwardTimeMajorResult(
            logits=torch.stack(logits_steps, dim=0),
            values=torch.stack(value_steps, dim=0),
        )

    def _score_public_heuristic_target_logits(
        self,
        *,
        forward_model: Any,
        obs_rows: Tensor,
        legal_actions: Any,
        observation_context: Mapping[str, Tensor] | None,
        device: torch.device,
    ) -> Tensor:
        profile_names = self._active_teacher_public_heuristic_profiles()
        if len(profile_names) > 1 and self.teacher_public_heuristic_profile_mode == "cycle":
            profile_names = (profile_names[int(self.update_count) % len(profile_names)],)
        profile_logits: list[Tensor] = []
        for profile_name in profile_names:
            profile_logits.append(
                torch.as_tensor(
                    forward_model.score_packed_public_heuristic_candidates(
                        obs_rows,
                        legal_actions,
                        observation_context=observation_context,
                        scoring_profile=profile_name,
                    ),
                    device=device,
                ).reshape(-1)
            )
        if not profile_logits:
            return torch.zeros((0,), device=device)
        if len(profile_logits) == 1:
            return profile_logits[0]
        offsets = torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long)
        row_count = max(int(offsets.shape[0]) - 1, 0)
        total_candidates = int(offsets[-1].item()) if offsets.numel() > 0 else 0
        if row_count == 0 or total_candidates == 0:
            return profile_logits[0]
        widths = (offsets[1:] - offsets[:-1]).to(dtype=torch.long)
        row_indices = torch.repeat_interleave(
            torch.arange(row_count, device=device, dtype=torch.long),
            widths,
        )
        scaled_profile_log_probs: list[Tensor] = []
        temperature = float(self.teacher_public_heuristic_temperature)
        for logits in profile_logits:
            scaled_logits = logits.to(device=device) / temperature
            row_log_z = _segment_logsumexp(scaled_logits, row_indices, row_count)
            scaled_profile_log_probs.append(scaled_logits - row_log_z.index_select(0, row_indices))
        mixture_log_probs = torch.logsumexp(
            torch.stack(scaled_profile_log_probs, dim=0),
            dim=0,
        ) - math.log(float(len(scaled_profile_log_probs)))
        return mixture_log_probs * temperature

    def _active_teacher_public_heuristic_profiles(self) -> tuple[str, ...]:
        profiles = self.teacher_public_heuristic_profiles
        if not profiles:
            return ("base",)
        end_updates = int(self.teacher_public_heuristic_profiles_end_updates)
        if end_updates >= 0 and int(self.update_count) > end_updates:
            return (profiles[0],)
        return profiles

    def _teacher_public_heuristic_rows(
        self,
        batch: Any,
        *,
        loss_mask: Tensor,
        expected_shape: tuple[int, int],
        action_catalog: ActionCatalog,
    ) -> Tensor:
        flat_loss_mask = loss_mask.reshape(-1) > 0.0
        teacher_valid = self._optional_time_major_bool_field(
            _batch_value(batch, "teacher_valid"),
            field_name="teacher_valid",
            expected_shape=expected_shape,
        )
        if teacher_valid is None:
            return torch.zeros((0,), dtype=torch.long, device=loss_mask.device)
        active_rows = flat_loss_mask & teacher_valid.reshape(-1).to(device=loss_mask.device, dtype=torch.bool)
        requested_families = tuple(self.teacher_public_heuristic_families)
        if requested_families:
            teacher_family = self._optional_time_major_index_field(
                _batch_value(batch, "teacher_family"),
                field_name="teacher_family",
                expected_shape=expected_shape,
            )
            if teacher_family is None:
                return torch.zeros((0,), dtype=torch.long, device=loss_mask.device)
            catalog_metadata = _structured_catalog_metadata(action_catalog)
            family_ids = _resolve_public_heuristic_family_ids(
                family_names=catalog_metadata.family_names,
                requested_families=requested_families,
            )
            if not family_ids:
                return torch.zeros((0,), dtype=torch.long, device=loss_mask.device)
            active_rows = active_rows & torch.isin(
                teacher_family.reshape(-1).to(device=loss_mask.device, dtype=torch.long),
                torch.as_tensor(family_ids, dtype=torch.long, device=loss_mask.device),
            )
        return torch.nonzero(active_rows, as_tuple=False).squeeze(1).to(dtype=torch.long)

    def _packed_public_heuristic_target_logits(
        self,
        *,
        forward_model: Any,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        observation_context: Mapping[str, Tensor] | None,
        active_rows: Tensor | None = None,
    ) -> Tensor | None:
        total_rows = int(obs.shape[0] * obs.shape[1])
        if active_rows is None:
            active_rows = torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        else:
            active_rows = active_rows.reshape(-1).to(device=obs.device, dtype=torch.long)
        if active_rows.numel() == 0:
            return None
        flat_obs = obs.reshape(total_rows, obs.shape[-1])
        if int(active_rows.shape[0]) == total_rows:
            legal_actions = self._packed_legal_action_view(packed_legal)
            return self._score_public_heuristic_target_logits(
                forward_model=forward_model,
                obs_rows=flat_obs,
                legal_actions=legal_actions,
                observation_context=observation_context,
                device=flat_obs.device,
            )
        subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
        subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
        subset_obs = flat_obs.index_select(0, active_rows)
        subset_context = (
            None
            if observation_context is None
            else self._subset_observation_context_rows(
                observation_context,
                active_rows,
                row_count=total_rows,
            )
        )
        subset_target_logits = self._score_public_heuristic_target_logits(
            forward_model=forward_model,
            obs_rows=subset_obs,
            legal_actions=subset_legal_actions,
            observation_context=subset_context,
            device=subset_obs.device,
        )
        return self._scatter_packed_candidate_values(
            packed_legal,
            active_rows,
            subset_target_logits,
            fill_value=0.0,
        )

    def _factorized_public_heuristic_teacher_view(
        self,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None],
        active_rows: Tensor | None = None,
    ) -> tuple[_PackedStructuredLegalView, Tensor] | tuple[None, None]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model for factorized public-heuristic distillation")
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not (
            hasattr(forward_model, "forward_trunk_sequence_seat_aware")
            and hasattr(forward_model, "score_packed_legal_candidates")
            and hasattr(forward_model, "score_packed_public_heuristic_candidates")
        ):
            return None, None

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        total_rows = int(expected_shape[0] * expected_shape[1])
        if active_rows is None:
            active_rows = torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        else:
            active_rows = active_rows.reshape(-1).to(device=obs.device, dtype=torch.long)
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

        student_started = time.perf_counter()
        recurrent_flat, state_repr, observation_context, _values, _seat_hidden = (
            forward_model.forward_trunk_sequence_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
            )
        )

        if int(active_rows.shape[0]) == total_rows:
            legal_actions_view = self._packed_legal_action_view(packed_legal)
            student_subset_logits = torch.as_tensor(
                forward_model.score_packed_legal_candidates(
                    recurrent_flat,
                    flat_obs,
                    legal_actions_view,
                    state_repr=state_repr,
                    observation_context=observation_context,
                    scoring_mode="learner",
                ),
                device=recurrent_flat.device,
            )
            self._record_timing_ms("learner_public_heuristic_student", time.perf_counter() - student_started)

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
            student_subset_logits = torch.as_tensor(
                forward_model.score_packed_legal_candidates(
                    recurrent_flat.index_select(0, active_rows),
                    subset_obs,
                    subset_legal_actions,
                    state_repr=state_repr.index_select(0, active_rows),
                    observation_context=subset_context,
                    scoring_mode="learner",
                ),
                device=recurrent_flat.device,
            )
            self._record_timing_ms("learner_public_heuristic_student", time.perf_counter() - student_started)

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
            target_logits = self._scatter_packed_candidate_values(
                packed_legal,
                active_rows,
                target_subset_logits,
                fill_value=0.0,
            )

        packed_view = _packed_structured_legal_view(
            logits=student_logits,
            packed_ids=packed_legal[0],
            packed_offsets=packed_legal[1],
            packed_meta=packed_legal[2],
        )
        return packed_view, target_logits

    def _should_use_factorized_legal_policy(
        self, forward_model: Any, *, packed_legal: tuple[Tensor, Tensor, Tensor | None] | None
    ) -> bool:
        return bool(
            packed_legal is not None
            and getattr(forward_model, "supports_factorized_legal_policy", False)
            and hasattr(forward_model, "evaluate_factorized_sequence_packed_seat_aware")
        )

    def _evaluate_factorized_time_major(
        self,
        batch: Any,
        *,
        obs: Tensor,
        actions: Tensor | None,
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
        active_rows = (
            None if loss_mask is None else torch.nonzero(loss_mask.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
        )
        same_family_reference_actions = None
        same_family_reference_families = None
        if float(self.teacher_same_family_action_coef) != 0.0 or float(self.teacher_action_coef) != 0.0:
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
        total_rows = int(expected_shape[0] * expected_shape[1])
        if active_rows is None or active_rows.numel() == 0 or int(active_rows.shape[0]) == total_rows:
            result = forward_model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
                legal_actions=self._packed_legal_action_view(packed_legal),
                actions=actions,
                same_family_reference_actions=same_family_reference_actions,
                same_family_reference_families=same_family_reference_families,
            )
        else:
            recurrent_flat, state_repr, observation_context, values, _seat_hidden = (
                forward_model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            )
            policy_head = forward_model.policy_head
            full_plan = policy_head._build_factorized_legality_plan(  # type: ignore[attr-defined]
                self._packed_legal_action_view(packed_legal),
                device=state_repr.device,
            )
            family_log_probs_full = policy_head._family_log_probs(state_repr, full_plan.family_mask)  # type: ignore[attr-defined]
            subset_packed_legal = self._slice_packed_legal_rows_with_meta(packed_legal, active_rows)
            subset_legal_actions = self._packed_legal_action_view(subset_packed_legal)
            flat_obs = obs.reshape(total_rows, obs.shape[-1])
            subset_result = policy_head.evaluate_factorized_packed(  # type: ignore[attr-defined]
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

            def _scatter_rows(values_subset: Tensor | None, *, fill_value: float = 0.0) -> Tensor | None:
                if values_subset is None:
                    return None
                full = values_subset.new_full((total_rows, *values_subset.shape[1:]), fill_value)
                full.index_copy_(0, active_rows, values_subset)
                return full

            subset_top_action_ids = subset_result.top_action_ids
            subset_same_family_action_logp = subset_result.same_family_action_logp
            subset_same_family_top_action_ids = subset_result.same_family_top_action_ids
            result = SimpleNamespace(
                values=values,
                action_logp=_scatter_rows(subset_result.action_logp),
                entropy=_scatter_rows(subset_result.entropy),
                family_log_probs=family_log_probs_full.reshape(
                    expected_shape[0], expected_shape[1], family_log_probs_full.shape[-1]
                ),
                play_slot_log_probs=(
                    None
                    if subset_result.play_slot_log_probs is None
                    else _scatter_rows(subset_result.play_slot_log_probs, fill_value=-torch.inf).reshape(
                        expected_shape[0],
                        expected_shape[1],
                        subset_result.play_slot_log_probs.shape[-1],
                    )
                ),
                move_slot_log_probs=(
                    None
                    if subset_result.move_slot_log_probs is None
                    else _scatter_rows(subset_result.move_slot_log_probs, fill_value=-torch.inf).reshape(
                        expected_shape[0],
                        expected_shape[1],
                        subset_result.move_slot_log_probs.shape[-1],
                    )
                ),
                attack_slot_log_probs=(
                    None
                    if subset_result.attack_slot_log_probs is None
                    else _scatter_rows(subset_result.attack_slot_log_probs, fill_value=-torch.inf).reshape(
                        expected_shape[0],
                        expected_shape[1],
                        subset_result.attack_slot_log_probs.shape[-1],
                    )
                ),
                attack_type_log_probs=(
                    None
                    if subset_result.attack_type_log_probs is None
                    else _scatter_rows(subset_result.attack_type_log_probs, fill_value=-torch.inf).reshape(
                        expected_shape[0],
                        expected_shape[1],
                        subset_result.attack_type_log_probs.shape[-1],
                    )
                ),
                top_action_ids=(
                    None
                    if subset_top_action_ids is None
                    else _scatter_rows(subset_top_action_ids, fill_value=-1).reshape(expected_shape)
                ),
                same_family_action_logp=(
                    None
                    if subset_same_family_action_logp is None
                    else _scatter_rows(subset_same_family_action_logp, fill_value=-torch.inf).reshape(
                        expected_shape,
                    )
                ),
                same_family_top_action_ids=(
                    None
                    if subset_same_family_top_action_ids is None
                    else _scatter_rows(subset_same_family_top_action_ids, fill_value=-1).reshape(
                        expected_shape,
                    )
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

    def _write_checkpoint_metadata(self) -> None:
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata_path = self.checkpoint_dir / f"checkpoint_metadata_{self.update_count}.json"
        checkpoint_metadata_path.write_text(
            json.dumps(
                {
                    "format": "checkpoint_metadata",
                    "parameters_included": False,
                    "update_count": self.update_count,
                    "policy_version": self.policy_version,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint metadata: {checkpoint_metadata_path}")

    def _log_metrics(self, update_metrics: dict[str, float], batch: Any) -> None:
        if not self.logger:
            return

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        vtrace_metrics = compute_vtrace_metrics(
            batch,
            rho_bar=self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
            c_bar=self.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = TrainingMetrics(
            update_count=self.update_count,
            wall_clock_seconds=elapsed,
            wall_clock_ms=int(elapsed * 1000),
            policy_version=self.policy_version,
            loss=float(update_metrics.get("loss", 0.0)),
            throughput_samples_per_sec=float(update_metrics.get("throughput_samples_per_sec", 0.0)),
            throughput_updates_per_sec=float(update_metrics.get("throughput_updates_per_sec", 0.0)),
            vtrace_rho_mean=vtrace_metrics.rho_mean,
            vtrace_rho_p50=float(update_metrics.get("vtrace_rho_p50", vtrace_metrics.rho_p50)),
            vtrace_rho_p90=float(update_metrics.get("vtrace_rho_p90", vtrace_metrics.rho_p90)),
            vtrace_rho_p99=float(update_metrics.get("vtrace_rho_p99", vtrace_metrics.rho_p99)),
            vtrace_clip_rate=float(update_metrics.get("vtrace_rho_clip_rate", vtrace_metrics.clip_rate)),
            vtrace_c_clipped_rate=float(update_metrics.get("vtrace_c_clip_rate", vtrace_metrics.c_clipped_rate)),
            kl_divergence=vtrace_metrics.kl_divergence,
            value_loss=float(update_metrics.get("value_loss", 0.0)),
            actor_loss=float(update_metrics.get("policy_loss", 0.0)),
            entropy=float(update_metrics.get("entropy", vtrace_metrics.entropy)),
            custom_metrics=self._custom_log_metrics(update_metrics, vtrace_metrics),
        )
        self.logger.log(metrics)

    def _custom_log_metrics(
        self,
        update_metrics: dict[str, float],
        vtrace_metrics: VtraceMetrics,
    ) -> dict[str, float]:
        return build_custom_log_metrics(update_metrics, vtrace_metrics)

    def get_policy_version(self) -> int:
        return self.policy_version
