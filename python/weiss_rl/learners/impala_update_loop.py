"""Optimizer-step orchestration for :class:`weiss_rl.learners.impala_learner.ImpalaLearner`."""

from __future__ import annotations

import time
from typing import Any, cast

import torch
from torch.nn.utils import clip_grad_norm_

from weiss_rl.learners.update_bookkeeping import throughput_metrics
from weiss_rl.learners.vtrace import VTraceTargets


def _batch_value(batch: Any, key: str) -> Any:
    # Resolve through impala_learner so the legacy helper remains the single compatibility hook.
    from weiss_rl.learners import impala_learner as learner_module

    return learner_module._batch_value(batch, key)


def _summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    # Resolve through impala_learner so the historical wrapper identity is preserved.
    from weiss_rl.learners import impala_learner as learner_module

    return learner_module.summarize_vtrace_diagnostics(result, rho_bar=rho_bar, c_bar=c_bar)


def _optimizer_has_gradients(optimizer: torch.optim.Optimizer) -> bool:
    return any(parameter.grad is not None for group in optimizer.param_groups for parameter in group.get("params", ()))


class ImpalaUpdateLoopMixin:
    def update(self: Any, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
        update_started = time.perf_counter()
        self.update_count += 1
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size

        elapsed = time.time() - self.start_time
        throughput_samples_per_sec, throughput_updates_per_sec = throughput_metrics(
            total_samples_processed=self.total_samples_processed,
            update_count=self.update_count,
            elapsed_seconds=elapsed,
        )

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
            self._active_timing_metrics = cast(dict[str, float] | None, {})
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
            if loss.requires_grad:
                if self._grad_scaler is not None:
                    loss_scale_before = float(self._grad_scaler.get_scale())
                    self._grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
            self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
            optimizer_started = time.perf_counter()
            if not loss.requires_grad:
                metrics.update(loss_metrics)
                metrics["optimizer_no_grad"] = 1.0
                metrics["amp_grad_overflow"] = 0.0
                metrics["loss_scale"] = 0.0 if self._grad_scaler is None else float(self._grad_scaler.get_scale())
                metrics["grad_norm"] = 0.0
            else:
                has_gradients = _optimizer_has_gradients(optimizer)
                if not has_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    metrics.update(loss_metrics)
                    metrics["optimizer_no_grad"] = 1.0
                    metrics["amp_grad_overflow"] = 0.0
                    metrics["loss_scale"] = 0.0 if loss_scale_before is None else float(loss_scale_before)
                    metrics["grad_norm"] = 0.0
                elif self._grad_scaler is not None:
                    self._grad_scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
                    gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
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
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
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
            metrics.update(_summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))

        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(metrics, batch)
            self.last_log_time = time.time()
            self.last_log_update = self.update_count

        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def auxiliary_update(self: Any, batch: Any) -> dict[str, float]:
        """Run one optimizer step using only structured teacher supervision."""
        update_started = time.perf_counter()
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run an auxiliary optimizer step")
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size
        if self.profile_timers:
            self._active_timing_metrics = cast(dict[str, float] | None, {})
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
        loss_scale_before = None
        if loss.requires_grad:
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
        optimizer_started = time.perf_counter()
        has_gradients = loss.requires_grad and _optimizer_has_gradients(optimizer)
        if not has_gradients:
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(aux_metrics)
            metrics["optimizer_no_grad"] = 1.0
            metrics["grad_norm"] = 0.0
            metrics["amp_grad_overflow"] = 0.0
            metrics["loss_scale"] = (
                0.0
                if self._grad_scaler is None
                else float(self._grad_scaler.get_scale() if loss_scale_before is None else loss_scale_before)
            )
        elif self._grad_scaler is not None:
            self._grad_scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
            gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
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
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
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

    def paired_swing_update(
        self: Any,
        batch: Any,
        *,
        margin: float,
        coef: float,
        positive_action_source: str,
        negative_action_source: str,
        loss_scope: str = "row",
        compare_to: str = "negative",
        margin_retention_coef: float = 0.0,
        margin_retention_margin: float = 0.0,
        top_action_retention_coef: float = 0.0,
        top_action_retention_margin: float = 0.0,
        full_surface_retention_batch: Any | None = None,
        full_surface_top_action_retention_coef: float = 0.0,
        full_surface_top_action_retention_margin: float = 0.0,
        full_surface_top_action_retention_mode: str = "reference_top",
    ) -> dict[str, float]:
        """Run one optimizer step using paired action-margin replay supervision."""

        update_started = time.perf_counter()
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run a paired-swing optimizer step")
        if float(full_surface_top_action_retention_coef) < 0.0:
            raise ValueError("full_surface_top_action_retention_coef must be >= 0")
        if float(full_surface_top_action_retention_margin) < 0.0:
            raise ValueError("full_surface_top_action_retention_margin must be >= 0")
        if float(full_surface_top_action_retention_coef) != 0.0 and full_surface_retention_batch is None:
            raise ValueError("full_surface_retention_batch is required when full-surface retention is active")
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size
        if self.profile_timers:
            self._active_timing_metrics = cast(dict[str, float] | None, {})
        self.model.train()
        if self.compiled_model is not None:
            self.compiled_model.train()
        loss_started = time.perf_counter()
        with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
            loss, swing_metrics, swing_context = self._paired_swing_loss_and_metrics(
                batch,
                margin=float(margin),
                coef=float(coef),
                positive_action_source=str(positive_action_source),
                negative_action_source=str(negative_action_source),
                loss_scope=str(loss_scope),
                compare_to=str(compare_to),
                margin_retention_coef=float(margin_retention_coef),
                margin_retention_margin=float(margin_retention_margin),
                top_action_retention_coef=float(top_action_retention_coef),
                top_action_retention_margin=float(top_action_retention_margin),
            )
            if full_surface_retention_batch is not None and float(full_surface_top_action_retention_coef) != 0.0:
                retention_loss, retention_metrics, retention_context = (
                    self._paired_swing_full_surface_top_action_retention_loss_and_metrics(
                        full_surface_retention_batch,
                        coef=float(full_surface_top_action_retention_coef),
                        margin=float(full_surface_top_action_retention_margin),
                        mode=str(full_surface_top_action_retention_mode),
                    )
                )
                loss = loss + retention_loss
                swing_metrics.update(retention_metrics)
                swing_context.update(retention_context)
        self._record_timing_ms("learner_paired_swing_loss_and_metrics", time.perf_counter() - loss_started)
        optimizer = self._optimizer_for_step()
        backward_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_scale_before = None
        if loss.requires_grad:
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
        optimizer_started = time.perf_counter()
        has_gradients = loss.requires_grad and _optimizer_has_gradients(optimizer)
        if not has_gradients:
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(swing_metrics)
            metrics["optimizer_no_grad"] = 1.0
            metrics["grad_norm"] = 0.0
            metrics["amp_grad_overflow"] = 0.0
            metrics["loss_scale"] = (
                0.0
                if self._grad_scaler is None
                else float(self._grad_scaler.get_scale() if loss_scale_before is None else loss_scale_before)
            )
        elif self._grad_scaler is not None:
            self._grad_scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
            gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
            if gradients_finite:
                self._grad_scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            self._grad_scaler.update()
            metrics = dict(swing_metrics)
            metrics["grad_norm"] = float(grad_norm_tensor)
            metrics["amp_grad_overflow"] = 0.0 if gradients_finite else 1.0
            metrics["loss_scale"] = float(self._grad_scaler.get_scale())
        else:
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self._ensure_finite_gradients(batch=batch, context=swing_context, grad_norm=grad_norm)
            optimizer.step()
            metrics = dict(swing_metrics)
            metrics["grad_norm"] = float(grad_norm)
        self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)
        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def paired_outcome_preference_update(
        self: Any,
        batch: Any,
        *,
        beta: float,
        coef: float,
        aggregation: str = "mean",
        group_balance: bool = False,
        retention_coef: float = 0.0,
        retention_margin: float = 0.0,
        retention_role: str = "preferred",
        retention_reference_top_only: bool = False,
        top_action_retention_coef: float = 0.0,
        top_action_retention_margin: float = 0.0,
        top_action_retention_role: str = "all",
        top_action_retention_reference_top_only: bool = False,
    ) -> dict[str, float]:
        """Run one optimizer step using paired trajectory/span preference replay."""

        update_started = time.perf_counter()
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run a paired outcome preference optimizer step")
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size
        if self.profile_timers:
            self._active_timing_metrics = cast(dict[str, float] | None, {})
        self.model.train()
        if self.compiled_model is not None:
            self.compiled_model.train()
        loss_started = time.perf_counter()
        with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
            loss, preference_metrics, preference_context = self._paired_outcome_preference_loss_and_metrics(
                batch,
                beta=float(beta),
                coef=float(coef),
                aggregation=str(aggregation),
                group_balance=bool(group_balance),
                retention_coef=float(retention_coef),
                retention_margin=float(retention_margin),
                retention_role=str(retention_role),
                retention_reference_top_only=bool(retention_reference_top_only),
                top_action_retention_coef=float(top_action_retention_coef),
                top_action_retention_margin=float(top_action_retention_margin),
                top_action_retention_role=str(top_action_retention_role),
                top_action_retention_reference_top_only=bool(top_action_retention_reference_top_only),
            )
        self._record_timing_ms("learner_paired_outcome_preference_loss_and_metrics", time.perf_counter() - loss_started)
        optimizer = self._optimizer_for_step()
        backward_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_scale_before = None
        if loss.requires_grad:
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
        optimizer_started = time.perf_counter()
        has_gradients = loss.requires_grad and _optimizer_has_gradients(optimizer)
        if not has_gradients:
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(preference_metrics)
            metrics["optimizer_no_grad"] = 1.0
            metrics["grad_norm"] = 0.0
            metrics["amp_grad_overflow"] = 0.0
            metrics["loss_scale"] = (
                0.0
                if self._grad_scaler is None
                else float(self._grad_scaler.get_scale() if loss_scale_before is None else loss_scale_before)
            )
        elif self._grad_scaler is not None:
            self._grad_scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
            gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
            if gradients_finite:
                self._grad_scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            self._grad_scaler.update()
            metrics = dict(preference_metrics)
            metrics["grad_norm"] = float(grad_norm_tensor)
            metrics["amp_grad_overflow"] = 0.0 if gradients_finite else 1.0
            metrics["loss_scale"] = float(self._grad_scaler.get_scale())
        else:
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self._ensure_finite_gradients(batch=batch, context=preference_context, grad_norm=grad_norm)
            optimizer.step()
            metrics = dict(preference_metrics)
            metrics["grad_norm"] = float(grad_norm)
        self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)
        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics
