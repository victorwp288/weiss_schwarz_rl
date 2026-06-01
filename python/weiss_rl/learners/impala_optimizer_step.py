"""Shared optimizer-step mechanics for IMPALA update paths."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_


def optimizer_has_gradients(optimizer: torch.optim.Optimizer) -> bool:
    return any(parameter.grad is not None for group in optimizer.param_groups for parameter in group.get("params", ()))


def run_impala_optimizer_step(
    *,
    learner: Any,
    batch: Any,
    loss: Tensor,
    base_metrics: dict[str, float],
    context: dict[str, Any],
    scale_loss_on_nonfinite_gradients: bool,
) -> dict[str, float]:
    optimizer = learner._optimizer_for_step()
    backward_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss_scale_before = None
    if loss.requires_grad:
        if learner._grad_scaler is not None:
            loss_scale_before = float(learner._grad_scaler.get_scale())
            learner._grad_scaler.scale(loss).backward()
        else:
            loss.backward()
    learner._record_timing_ms("learner_backward", time.perf_counter() - backward_started)

    optimizer_started = time.perf_counter()
    metrics = dict(base_metrics)
    if not loss.requires_grad:
        metrics["optimizer_no_grad"] = 1.0
        metrics["amp_grad_overflow"] = 0.0
        metrics["loss_scale"] = 0.0 if learner._grad_scaler is None else float(learner._grad_scaler.get_scale())
        metrics["grad_norm"] = 0.0
    else:
        has_gradients = optimizer_has_gradients(optimizer)
        if not has_gradients:
            optimizer.zero_grad(set_to_none=True)
            metrics["optimizer_no_grad"] = 1.0
            metrics["amp_grad_overflow"] = 0.0
            metrics["loss_scale"] = 0.0 if loss_scale_before is None else float(loss_scale_before)
            metrics["grad_norm"] = 0.0
        elif learner._grad_scaler is not None:
            _apply_scaled_optimizer_step(
                learner=learner,
                optimizer=optimizer,
                metrics=metrics,
                loss_scale_before=loss_scale_before,
                scale_loss_on_nonfinite_gradients=scale_loss_on_nonfinite_gradients,
            )
        else:
            grad_norm = clip_grad_norm_(learner.model.parameters(), learner.grad_norm_clip)
            learner._ensure_finite_gradients(batch=batch, context=context, grad_norm=grad_norm)
            optimizer.step()
            metrics["grad_norm"] = float(grad_norm)
    learner._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)
    return metrics


def _apply_scaled_optimizer_step(
    *,
    learner: Any,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    loss_scale_before: float | None,
    scale_loss_on_nonfinite_gradients: bool,
) -> None:
    learner._grad_scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(learner.model.parameters(), learner.grad_norm_clip)
    bad_gradients, grad_norm_tensor = learner._collect_nonfinite_gradients(grad_norm)
    gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
    if gradients_finite:
        learner._grad_scaler.step(optimizer)
        learner._grad_scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        if scale_loss_on_nonfinite_gradients and loss_scale_before is not None:
            try:
                learner._grad_scaler.update(loss_scale_before * 0.5)
            except TypeError:
                learner._grad_scaler.update()
        else:
            learner._grad_scaler.update()
    loss_scale_after = float(learner._grad_scaler.get_scale())
    gradient_overflow = (not gradients_finite) or bool(
        scale_loss_on_nonfinite_gradients and loss_scale_before is not None and loss_scale_after < loss_scale_before
    )
    metrics["grad_norm"] = float(grad_norm_tensor)
    metrics["amp_grad_overflow"] = 1.0 if gradient_overflow else 0.0
    metrics["loss_scale"] = loss_scale_after


__all__ = ["optimizer_has_gradients", "run_impala_optimizer_step"]
