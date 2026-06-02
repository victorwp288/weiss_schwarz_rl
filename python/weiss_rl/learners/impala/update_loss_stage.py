"""Shared IMPALA update loss-building stage."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.impala.update_bookkeeping import set_impala_model_train_mode

ScopedLossBuilder = Callable[[], tuple[Tensor, dict[str, float], dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ScopedImpalaLossBuild:
    loss: Tensor
    metrics: dict[str, float]
    context: dict[str, Any]


def build_scoped_impala_loss(
    *,
    learner: Any,
    loss_timer_name: str,
    build_loss: ScopedLossBuilder,
) -> ScopedImpalaLossBuild:
    set_impala_model_train_mode(learner)
    loss_started = time.perf_counter()
    with torch.amp.autocast(device_type=learner._amp_device_type, enabled=learner._amp_enabled):
        loss, metrics, context = build_loss()
    learner._record_timing_ms(loss_timer_name, time.perf_counter() - loss_started)
    return ScopedImpalaLossBuild(loss=loss, metrics=metrics, context=context)


__all__ = ["ScopedImpalaLossBuild", "ScopedLossBuilder", "build_scoped_impala_loss"]
