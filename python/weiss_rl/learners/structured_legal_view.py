"""Packed legal-action views used by structured auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import segment_logsumexp, segment_max


@dataclass(frozen=True, slots=True)
class PackedStructuredLegalView:
    row_count: int
    row_indices: Tensor
    action_ids: Tensor
    logits: Tensor
    row_log_z: Tensor
    row_has_candidates: Tensor
    family_ids: Tensor
    arg0: Tensor
    arg1: Tensor
    arg2: Tensor


def packed_structured_legal_view(
    *,
    logits: Tensor | None,
    packed_ids: Tensor | None,
    packed_offsets: Tensor | None,
    packed_meta: Tensor | None,
) -> PackedStructuredLegalView | None:
    """Build a row-indexed view over packed legal actions."""
    if packed_ids is None or packed_offsets is None or packed_meta is None:
        return None
    if logits is None:
        selected_logits = torch.zeros((int(packed_ids.shape[0]),), device=packed_ids.device, dtype=torch.float32)
        row_count = int(packed_offsets.shape[0] - 1)
        flat_device = packed_ids.device
    elif logits.ndim == 1:
        selected_logits = logits.to(dtype=torch.float32)
        row_count = int(packed_offsets.shape[0] - 1)
        flat_device = selected_logits.device
        if int(selected_logits.shape[0]) != int(packed_ids.shape[0]):
            raise ValueError("packed logits must align 1:1 with packed ids")
    else:
        flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
        row_count = int(flat_logits.shape[0])
        flat_device = flat_logits.device
        lengths = packed_offsets[1:] - packed_offsets[:-1]
        if lengths.ndim != 1 or lengths.numel() != row_count:
            raise ValueError(f"packed legal offsets must describe {row_count} rows")
        row_indices = torch.repeat_interleave(
            torch.arange(row_count, device=flat_device, dtype=torch.long),
            lengths.to(device=flat_device, dtype=torch.long),
        )
        selected_logits = (
            flat_logits[row_indices, packed_ids.to(device=flat_device, dtype=torch.long)]
            if row_indices.numel() > 0
            else flat_logits.new_zeros((0,))
        )
    lengths = packed_offsets[1:] - packed_offsets[:-1]
    if lengths.ndim != 1 or lengths.numel() != row_count:
        raise ValueError(f"packed legal offsets must describe {row_count} rows")
    if packed_meta.ndim != 2 or int(packed_meta.shape[0]) != int(packed_ids.shape[0]) or int(packed_meta.shape[1]) < 4:
        raise ValueError("packed legal metadata must align 1:1 with packed ids and expose 4 fields")
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, device=flat_device, dtype=torch.long),
        lengths.to(device=flat_device, dtype=torch.long),
    )
    meta_long = packed_meta.to(device=flat_device, dtype=torch.long)
    unused = int(np.iinfo(np.uint16).max)

    def _normalize_meta_column(column: Tensor) -> Tensor:
        return torch.where(column == unused, torch.full_like(column, -1), column)

    return PackedStructuredLegalView(
        row_count=row_count,
        row_indices=row_indices,
        action_ids=packed_ids.to(device=flat_device, dtype=torch.long),
        logits=selected_logits,
        row_log_z=segment_logsumexp(selected_logits, row_indices, row_count),
        row_has_candidates=lengths.to(device=flat_device, dtype=torch.bool),
        family_ids=_normalize_meta_column(meta_long[:, 0]),
        arg0=_normalize_meta_column(meta_long[:, 1]),
        arg1=_normalize_meta_column(meta_long[:, 2]),
        arg2=_normalize_meta_column(meta_long[:, 3]),
    )


def packed_group_log_probs(
    packed_view: PackedStructuredLegalView,
    *,
    group_ids: Tensor,
    group_count: int,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Compute per-row log probability mass for packed action groups."""
    group_count = int(group_count)
    out = torch.full(
        (packed_view.row_count, max(group_count, 1)),
        -torch.inf,
        dtype=packed_view.logits.dtype,
        device=packed_view.logits.device,
    )[:, :group_count]
    if group_count <= 0 or packed_view.logits.numel() == 0:
        return out
    selected = (
        torch.ones_like(group_ids, dtype=torch.bool) if candidate_mask is None else candidate_mask.to(dtype=torch.bool)
    )
    row_log_z = (
        packed_view.row_log_z
        if candidate_mask is None
        else segment_logsumexp(
            packed_view.logits[selected],
            packed_view.row_indices[selected],
            packed_view.row_count,
        )
    )
    valid = selected & (group_ids >= 0) & (group_ids < group_count)
    if not bool(valid.any().item()):
        return out
    flat_keys = packed_view.row_indices[valid].to(dtype=torch.long) * group_count + group_ids[valid].to(
        dtype=torch.long
    )
    grouped = segment_logsumexp(packed_view.logits[valid], flat_keys, packed_view.row_count * group_count).view(
        packed_view.row_count,
        group_count,
    )
    finite_rows = torch.isfinite(row_log_z)
    if bool(finite_rows.any().item()):
        out[finite_rows] = grouped[finite_rows] - row_log_z[finite_rows].unsqueeze(1)
    return out


def packed_soft_target_cross_entropy(
    packed_view: PackedStructuredLegalView,
    *,
    target_logits: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-row soft-target cross entropy, top mass, and entropy."""
    if temperature <= 0.0:
        raise ValueError("public heuristic temperature must be > 0")
    flat_target_logits = target_logits.reshape(-1).to(device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    if int(flat_target_logits.shape[0]) != int(packed_view.logits.shape[0]):
        raise ValueError("public heuristic target logits must align 1:1 with packed logits")
    scaled_target_logits = flat_target_logits / float(temperature)
    target_row_log_z = segment_logsumexp(scaled_target_logits, packed_view.row_indices, packed_view.row_count)
    target_log_probs = scaled_target_logits - target_row_log_z.index_select(
        0, packed_view.row_indices.to(dtype=torch.long)
    )
    target_probs = torch.exp(target_log_probs)
    student_log_probs = packed_view.logits - packed_view.row_log_z.index_select(
        0, packed_view.row_indices.to(dtype=torch.long)
    )

    row_cross_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    row_cross_entropy.scatter_add_(
        0,
        packed_view.row_indices.to(dtype=torch.long),
        -(target_probs * student_log_probs),
    )

    row_target_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    row_target_entropy.scatter_add_(
        0,
        packed_view.row_indices.to(dtype=torch.long),
        -(target_probs * target_log_probs),
    )

    student_top_logits = segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
    student_top_mask = packed_view.logits >= (
        student_top_logits.index_select(0, packed_view.row_indices.to(dtype=torch.long)) - 1.0e-6
    )
    row_student_top_mass = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    if bool(student_top_mask.any().item()):
        row_student_top_mass.scatter_add_(
            0,
            packed_view.row_indices[student_top_mask].to(dtype=torch.long),
            target_probs[student_top_mask],
        )
    return row_cross_entropy, row_student_top_mass, row_target_entropy


__all__ = [
    "PackedStructuredLegalView",
    "packed_group_log_probs",
    "packed_soft_target_cross_entropy",
    "packed_structured_legal_view",
]
