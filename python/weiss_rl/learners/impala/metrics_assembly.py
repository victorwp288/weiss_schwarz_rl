"""IMPALA learner metric assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala.loss_metrics import build_impala_loss_metrics
from weiss_rl.learners.impala.structured_summary import (
    DenseMaskResolver,
    ImpalaStructuredSummaryRequest,
    TimingRecorder,
    compute_impala_structured_policy_summary,
)
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView

BatchValueGetter = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class ImpalaMetricAssemblyRequest:
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_mean: Tensor
    entropy_scope: str
    loss_mask: Tensor
    value_loss_mask: Tensor
    actions: Tensor
    action_logp: Tensor
    behavior_logp_for_mask: Tensor | None
    rewards_for_metrics: Tensor
    advantages: Tensor
    targets: Tensor
    rhos_for_metrics: Tensor
    rho_bar: float
    c_bar: float
    action_catalog: Any
    pass_action_id: int | None
    trajectory_retention_metrics: dict[str, float] = field(default_factory=dict)
    policy_anchor_metrics: dict[str, float] = field(default_factory=dict)
    teacher_metrics: dict[str, float] = field(default_factory=dict)
    emit_structured_metrics: bool = False
    logits: Tensor | None = None
    legal_mask: Tensor | None = None
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None = None
    packed_view: PackedStructuredLegalView | None = None
    factorized_result: Any = None
    batch: Any = None
    expected_shape: torch.Size | None = None
    action_dim: int | None = None
    resolve_legal_mask: DenseMaskResolver | None = None


def assemble_impala_loss_metrics(
    request: ImpalaMetricAssemblyRequest,
    *,
    batch_value: BatchValueGetter,
    record_timing_ms: TimingRecorder,
) -> dict[str, float]:
    structured_action_catalog = request.action_catalog if isinstance(request.action_catalog, ActionCatalog) else None
    metrics = build_impala_loss_metrics(
        total_loss=request.total_loss,
        policy_loss=request.policy_loss,
        value_loss=request.value_loss,
        entropy_mean=request.entropy_mean,
        entropy_scope=request.entropy_scope,
        loss_mask=request.loss_mask,
        value_loss_mask=request.value_loss_mask,
        actions=request.actions,
        action_logp=request.action_logp,
        behavior_logp_for_mask=request.behavior_logp_for_mask,
        rewards_for_metrics=request.rewards_for_metrics,
        advantages=request.advantages,
        targets=request.targets,
        rhos_for_metrics=request.rhos_for_metrics,
        rho_bar=request.rho_bar,
        c_bar=request.c_bar,
        action_catalog=structured_action_catalog,
        pass_action_id=request.pass_action_id,
        terminal_outcome_backfill_count=batch_value(request.batch, "terminal_outcome_backfill_count"),
        terminal_outcome_backfill_total_micros=batch_value(request.batch, "terminal_outcome_backfill_total_micros"),
        terminal_outcome_trace_backfill_count=batch_value(request.batch, "terminal_outcome_trace_backfill_count"),
        terminal_outcome_trace_backfill_total_micros=batch_value(
            request.batch,
            "terminal_outcome_trace_backfill_total_micros",
        ),
        trajectory_retention_metrics=request.trajectory_retention_metrics,
        policy_anchor_metrics=request.policy_anchor_metrics,
        teacher_metrics=request.teacher_metrics,
    )
    if structured_action_catalog is not None and request.emit_structured_metrics:
        metrics.update(
            compute_impala_structured_policy_summary(
                ImpalaStructuredSummaryRequest(
                    logits=request.logits,
                    legal_mask=request.legal_mask,
                    action_catalog=structured_action_catalog,
                    packed_legal=request.packed_legal,
                    packed_view=request.packed_view,
                    factorized_result=request.factorized_result,
                    batch=request.batch,
                    expected_shape=request.expected_shape,
                    action_dim=request.action_dim,
                    resolve_legal_mask=request.resolve_legal_mask,
                ),
                record_timing_ms=record_timing_ms,
            )
        )
    return metrics


__all__ = ["ImpalaMetricAssemblyRequest", "assemble_impala_loss_metrics"]
