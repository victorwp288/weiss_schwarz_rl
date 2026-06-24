"""Packed legal-candidate opponent-context adjustments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.legal_actions import LegalActionBatch


def apply_packed_action_bias(
    owner: Any,
    packed_logits: Tensor,
    legal_actions: LegalActionBatch,
    opponent_context_index: Tensor | None,
) -> Tensor:
    adapter = getattr(owner, "opponent_context_action_bias_adapter", None)
    scale = float(getattr(owner, "opponent_context_trainable_action_bias_scale", 0.0))
    if adapter is None or scale == 0.0:
        return packed_logits
    if legal_actions.ids is None or legal_actions.offsets is None:
        return packed_logits
    if packed_logits.ndim != 1:
        raise ValueError(f"packed_logits must be 1D, got shape {tuple(packed_logits.shape)}")
    offsets = torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    indices = owner._opponent_context_indices_tensor(
        opponent_context_index,
        row_count=row_count,
        device=packed_logits.device,
        adapter_name="opponent_context_action_bias_adapter",
    )
    if indices is None:
        return packed_logits
    lengths = offsets[1:] - offsets[:-1]
    if int(lengths.sum().item()) != int(packed_logits.shape[0]):
        raise ValueError("packed legal offsets must align with packed logits")
    row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_logits.device), lengths)
    action_ids = torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long)
    if int(action_ids.numel()) != int(packed_logits.shape[0]):
        raise ValueError("packed legal ids must align with packed logits")
    row_context = indices.index_select(0, row_indices)
    bias_table = adapter.to(device=packed_logits.device, dtype=packed_logits.dtype)
    bias = bias_table[row_context, action_ids.clamp(min=0, max=int(bias_table.shape[1]) - 1)] * scale
    bias = bias.masked_fill(row_context == 0, 0.0)
    return packed_logits + bias


def apply_packed_candidate_residual(
    owner: Any,
    packed_logits: Tensor,
    legal_actions: LegalActionBatch,
    state_repr: Tensor,
    opponent_context_index: Tensor | None,
    *,
    observation_context: Mapping[str, Tensor] | None = None,
    scoring_mode: str = "auto",
) -> Tensor:
    context_table = getattr(owner, "opponent_context_candidate_residual_context", None)
    state_layer = getattr(owner, "opponent_context_candidate_residual_state", None)
    candidate_layer = getattr(owner, "opponent_context_candidate_residual_candidate", None)
    meta_layer = getattr(owner, "opponent_context_candidate_residual_meta", None)
    out_layer = getattr(owner, "opponent_context_candidate_residual_out", None)
    scale = float(getattr(owner, "opponent_context_trainable_candidate_residual_scale", 0.0))
    if context_table is None or state_layer is None or meta_layer is None or out_layer is None or scale == 0.0:
        return packed_logits
    if legal_actions.offsets is None or legal_actions.meta is None:
        return packed_logits
    if packed_logits.ndim != 1:
        raise ValueError(f"packed_logits must be 1D, got shape {tuple(packed_logits.shape)}")
    if state_repr.ndim != 2:
        raise ValueError(f"state_repr must be 2D, got shape {tuple(state_repr.shape)}")
    offsets = torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    if int(state_repr.shape[0]) != row_count:
        raise ValueError(f"state_repr row count must match packed offsets, got {int(state_repr.shape[0])}")
    indices = owner._opponent_context_indices_tensor(
        opponent_context_index,
        row_count=row_count,
        device=packed_logits.device,
        adapter_name="opponent_context_candidate_residual_context",
    )
    if indices is None:
        return packed_logits
    lengths = offsets[1:] - offsets[:-1]
    if int(lengths.sum().item()) != int(packed_logits.shape[0]):
        raise ValueError("packed legal offsets must align with packed logits")
    if int(packed_logits.shape[0]) == 0:
        return packed_logits
    row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_logits.device), lengths)
    row_context = indices.index_select(0, row_indices)
    raw_meta = torch.as_tensor(legal_actions.meta, device=packed_logits.device, dtype=torch.float32)
    if raw_meta.ndim != 2 or raw_meta.shape[0] != packed_logits.shape[0] or raw_meta.shape[1] < 3:
        raise ValueError("packed legal meta must have shape (packed_actions, >=3)")
    meta = raw_meta[:, :3]
    meta = torch.where(meta >= 60000.0, torch.full_like(meta, -1.0), meta)
    meta_scale = meta.new_tensor([32.0, 64.0, 64.0])
    meta_features = meta / meta_scale
    row_state = state_repr.to(device=packed_logits.device, dtype=packed_logits.dtype).index_select(0, row_indices)
    context_features = context_table.to(device=packed_logits.device, dtype=packed_logits.dtype).index_select(
        0,
        row_context,
    )
    residual = _candidate_residual_scores(
        owner,
        legal_actions,
        state_repr,
        packed_logits,
        row_state=row_state,
        context_features=context_features,
        meta_features=meta_features,
        state_layer=state_layer,
        candidate_layer=candidate_layer,
        meta_layer=meta_layer,
        out_layer=out_layer,
        observation_context=observation_context,
        scoring_mode=scoring_mode,
    )
    residual = residual * scale
    residual = _mask_unconfigured_candidate_residual_actions(owner, residual, legal_actions, packed_logits)
    residual = residual.masked_fill(row_context == 0, 0.0)
    return packed_logits + residual


def apply_packed_candidate_residual_to_log_probs(
    owner: Any,
    packed_log_probs: Tensor,
    legal_actions: LegalActionBatch,
    state_repr: Tensor,
    opponent_context_index: Tensor | None,
    *,
    observation_context: Mapping[str, Tensor] | None = None,
    scoring_mode: str = "auto",
) -> Tensor:
    if legal_actions.offsets is None or legal_actions.meta is None:
        return packed_log_probs
    biased = owner._apply_opponent_context_packed_candidate_residual(
        packed_log_probs,
        legal_actions,
        state_repr,
        opponent_context_index,
        observation_context=observation_context,
        scoring_mode=scoring_mode,
    )
    if biased is packed_log_probs:
        return packed_log_probs
    return normalize_packed_log_probs(biased, legal_actions, value_name="packed_log_probs")


def apply_packed_action_bias_to_log_probs(
    owner: Any,
    packed_log_probs: Tensor,
    legal_actions: LegalActionBatch,
    opponent_context_index: Tensor | None,
) -> Tensor:
    if legal_actions.ids is None or legal_actions.offsets is None:
        return packed_log_probs
    biased = owner._apply_opponent_context_packed_action_bias(
        packed_log_probs,
        legal_actions,
        opponent_context_index,
    )
    if biased is packed_log_probs:
        return packed_log_probs
    return normalize_packed_log_probs(biased, legal_actions, value_name="packed_log_probs")


def normalize_packed_log_probs(values: Tensor, legal_actions: LegalActionBatch, *, value_name: str) -> Tensor:
    if values.ndim != 1:
        raise ValueError(f"{value_name} must be 1D, got shape {tuple(values.shape)}")
    offsets = torch.as_tensor(legal_actions.offsets, device=values.device, dtype=torch.long)
    lengths = offsets[1:] - offsets[:-1]
    if int(lengths.sum().item()) != int(values.shape[0]):
        raise ValueError("packed legal offsets must align with packed log-probs")
    if int(values.shape[0]) == 0:
        return values
    row_count = int(offsets.numel() - 1)
    row_indices = torch.repeat_interleave(torch.arange(row_count, device=values.device), lengths)
    row_log_z = segment_logsumexp_1d(values, row_indices, row_count)
    return values - row_log_z.index_select(0, row_indices)


def segment_logsumexp_1d(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    out_max = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    if keys.numel() == 0:
        return out_max
    long_keys = keys.to(dtype=torch.long)
    out_max.scatter_reduce_(0, long_keys, values, reduce="amax", include_self=True)
    gathered_max = out_max.index_select(0, long_keys)
    shifted = torch.exp(values - gathered_max)
    sumexp = torch.zeros((int(num_segments),), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, long_keys, shifted)
    valid = torch.isfinite(out_max) & (sumexp > 0)
    result = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    result[valid] = torch.log(sumexp[valid]) + out_max[valid]
    return result


def _candidate_residual_scores(
    owner: Any,
    legal_actions: LegalActionBatch,
    state_repr: Tensor,
    packed_logits: Tensor,
    *,
    row_state: Tensor,
    context_features: Tensor,
    meta_features: Tensor,
    state_layer: Any,
    candidate_layer: Any,
    meta_layer: Any,
    out_layer: Any,
    observation_context: Mapping[str, Tensor] | None,
    scoring_mode: str,
) -> Tensor:
    mode = str(getattr(owner, "opponent_context_candidate_residual_mode", "additive")).strip().lower()
    if mode in {"rich", "rich_bilinear"}:
        if candidate_layer is None:
            raise ValueError("rich opponent-context candidate residual requires candidate residual layer")
        if observation_context is None:
            raise ValueError("rich opponent-context candidate residual requires observation_context")
        candidate_repr_fn = getattr(owner.policy_head, "_project_packed_candidate_representations", None)
        if not callable(candidate_repr_fn):
            raise ValueError("rich opponent-context candidate residual requires packed candidate representations")
        candidate_repr = candidate_repr_fn(
            state_repr.to(device=packed_logits.device, dtype=packed_logits.dtype),
            legal_actions,
            observation_context,
            scoring_mode=scoring_mode,
        ).to(device=packed_logits.device, dtype=packed_logits.dtype)
        if tuple(candidate_repr.shape[:1]) != tuple(packed_logits.shape[:1]):
            raise ValueError("rich candidate residual representation must align with packed logits")
        candidate_features = (
            state_layer(row_state)
            + candidate_layer(candidate_repr)
            + meta_layer(meta_features.to(dtype=packed_logits.dtype))
        )
        if mode == "rich_bilinear":
            hidden = torch.tanh(candidate_features)
            return (hidden * context_features).sum(dim=-1).to(dtype=packed_logits.dtype)
        hidden = torch.tanh(candidate_features + context_features)
        return out_layer(hidden).squeeze(-1).to(dtype=packed_logits.dtype)
    candidate_features = state_layer(row_state) + meta_layer(meta_features.to(dtype=packed_logits.dtype))
    if mode == "bilinear":
        hidden = torch.tanh(candidate_features)
        return (hidden * context_features).sum(dim=-1).to(dtype=packed_logits.dtype)
    hidden = torch.tanh(candidate_features + context_features)
    return out_layer(hidden).squeeze(-1).to(dtype=packed_logits.dtype)


def _mask_unconfigured_candidate_residual_actions(
    owner: Any,
    residual: Tensor,
    legal_actions: LegalActionBatch,
    packed_logits: Tensor,
) -> Tensor:
    allowed_action_ids = tuple(
        int(action_id) for action_id in getattr(owner, "opponent_context_candidate_residual_action_ids", ())
    )
    if not allowed_action_ids:
        return residual
    if legal_actions.ids is None:
        return torch.zeros_like(residual)
    action_ids = torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long)
    if int(action_ids.numel()) != int(packed_logits.shape[0]):
        raise ValueError("packed legal ids must align with packed logits")
    allowed = torch.zeros_like(action_ids, dtype=torch.bool)
    for action_id in allowed_action_ids:
        allowed |= action_ids == int(action_id)
    return residual.masked_fill(~allowed, 0.0)


__all__ = [
    "apply_packed_action_bias",
    "apply_packed_action_bias_to_log_probs",
    "apply_packed_candidate_residual",
    "apply_packed_candidate_residual_to_log_probs",
    "normalize_packed_log_probs",
    "segment_logsumexp_1d",
]
