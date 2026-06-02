"""IMPALA structured policy summary orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView
from weiss_rl.learners.structured_policy_metrics import summarize_structured_policy_metrics

TimingRecorder = Callable[[str, float], None]
DenseMaskResolver = Callable[[Any, torch.Size, int], Tensor]


@dataclass(frozen=True, slots=True)
class ImpalaStructuredSummaryRequest:
    logits: Tensor | None
    legal_mask: Tensor | None
    action_catalog: ActionCatalog
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None = None
    packed_view: PackedStructuredLegalView | None = None
    factorized_result: Any = None
    batch: Any = None
    expected_shape: torch.Size | None = None
    action_dim: int | None = None
    resolve_legal_mask: DenseMaskResolver | None = None


def compute_impala_structured_policy_summary(
    request: ImpalaStructuredSummaryRequest,
    *,
    record_timing_ms: TimingRecorder,
) -> dict[str, float]:
    structured_legal_mask = _resolve_structured_legal_mask(request)
    summary_started = time.perf_counter()
    metrics = summarize_structured_policy_metrics(
        request.logits,
        structured_legal_mask,
        action_catalog=request.action_catalog,
        packed_ids=None if request.packed_legal is None else request.packed_legal[0],
        packed_offsets=None if request.packed_legal is None else request.packed_legal[1],
        packed_meta=None if request.packed_legal is None else request.packed_legal[2],
        packed_view=request.packed_view,
        factorized_family_log_probs=None
        if request.factorized_result is None
        else request.factorized_result.family_log_probs,
    )
    record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
    return metrics


def _resolve_structured_legal_mask(request: ImpalaStructuredSummaryRequest) -> Tensor | None:
    if request.factorized_result is not None:
        return None
    if request.legal_mask is not None:
        return request.legal_mask
    if request.packed_legal is not None and request.packed_legal[2] is not None:
        return None
    if request.resolve_legal_mask is None:
        return None
    if request.expected_shape is None or request.action_dim is None:
        raise ValueError("dense structured summary fallback requires expected_shape and action_dim")
    return request.resolve_legal_mask(request.batch, request.expected_shape, request.action_dim)


__all__ = ["ImpalaStructuredSummaryRequest", "compute_impala_structured_policy_summary"]
