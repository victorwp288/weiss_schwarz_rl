"""Small learner update bookkeeping helpers."""

from __future__ import annotations

from typing import Any

import torch


def record_timing_ms(
    active_timing_metrics: dict[str, float] | None,
    *,
    profile_timers: bool,
    name: str,
    elapsed_seconds: float,
) -> None:
    """Accumulate a timing metric when learner profiling is enabled."""

    if not bool(profile_timers) or active_timing_metrics is None:
        return
    key = f"timer_{name}_ms"
    active_timing_metrics[key] = active_timing_metrics.get(key, 0.0) + (float(elapsed_seconds) * 1000.0)


def teacher_aux_active(*, teacher_aux_mode: str, auxiliary_update: bool) -> bool:
    """Return whether structured teacher auxiliary losses should be active."""

    mode = str(teacher_aux_mode)
    if mode == "off":
        return False
    if mode == "warmstart_only":
        return bool(auxiliary_update)
    return True


def should_emit_structured_metrics(
    *,
    structured_metrics_mode: str,
    auxiliary_update: bool,
    update_count: int,
) -> bool:
    """Return whether structured learner metrics should be emitted for this update."""

    mode = str(structured_metrics_mode)
    if mode == "off":
        return False
    if mode == "sampled":
        return (not bool(auxiliary_update)) and (int(update_count) % 10 == 0)
    return True


def throughput_metrics(
    *,
    total_samples_processed: int,
    update_count: int,
    elapsed_seconds: float,
) -> tuple[float, float]:
    """Return sample/sec and update/sec learner throughput metrics."""

    elapsed = max(float(elapsed_seconds), 1e-6)
    return int(total_samples_processed) / elapsed, int(update_count) / elapsed


def learner_acceleration_state(
    *,
    model: Any | None,
    mixed_precision: bool,
) -> tuple[bool, str, torch.amp.GradScaler | None]:
    """Resolve learner autocast and gradient-scaler state from the current model."""

    if model is None:
        return False, "cpu", None
    parameter = next(model.parameters(), None)
    if parameter is None:
        return False, "cpu", None
    amp_device_type = parameter.device.type
    amp_enabled = bool(mixed_precision and amp_device_type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    return amp_enabled, amp_device_type, grad_scaler
