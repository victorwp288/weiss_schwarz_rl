"""Policy-anchor regularization helpers."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from weiss_rl.learners.tensor_ops import segment_logsumexp, segment_max, weighted_mean


def clone_frozen_policy_anchor(model: nn.Module) -> nn.Module:
    """Return an eval-mode frozen copy of the current policy model."""

    anchor = copy.deepcopy(model)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor


def packed_candidate_anchor_kl_loss(
    *,
    current_log_probs: Tensor,
    anchor_log_probs: Tensor,
    packed_offsets: Tensor,
    row_shape: tuple[int, int],
    loss_mask: Tensor,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute KL(anchor || current) for packed legal-candidate log-probs."""

    temperature_f = float(temperature)
    if temperature_f <= 0.0:
        raise ValueError("policy anchor temperature must be > 0")
    if current_log_probs.shape != anchor_log_probs.shape:
        raise ValueError(
            "current and anchor packed log-prob tensors must have identical shapes, "
            f"got {tuple(current_log_probs.shape)} and {tuple(anchor_log_probs.shape)}"
        )
    offsets = packed_offsets.to(device=current_log_probs.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    expected_rows = int(row_shape[0] * row_shape[1])
    if row_count != expected_rows:
        raise ValueError(f"packed row count {row_count} does not match row_shape {row_shape}")
    if tuple(loss_mask.shape) != tuple(row_shape):
        raise ValueError(f"loss_mask must match row_shape {row_shape}, got {tuple(loss_mask.shape)}")

    lengths = offsets[1:] - offsets[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, dtype=torch.long, device=current_log_probs.device),
        lengths,
    )
    if int(row_indices.numel()) != int(current_log_probs.numel()):
        raise ValueError("packed offsets do not match packed log-prob tensor length")
    current_scores = current_log_probs.to(dtype=torch.float32) / temperature_f
    anchor_scores = anchor_log_probs.to(device=current_scores.device, dtype=torch.float32) / temperature_f
    current_log_z = segment_logsumexp(current_scores, row_indices, row_count)
    anchor_log_z = segment_logsumexp(anchor_scores, row_indices, row_count)
    current_logp = current_scores - current_log_z.index_select(0, row_indices)
    anchor_logp = anchor_scores - anchor_log_z.index_select(0, row_indices)
    anchor_prob = torch.exp(anchor_logp)
    candidate_kl = anchor_prob * (anchor_logp - current_logp)
    row_kl = torch.zeros((row_count,), dtype=torch.float32, device=current_scores.device)
    row_kl.scatter_add_(0, row_indices, candidate_kl)
    row_kl = row_kl.reshape(row_shape)
    mask = loss_mask.to(device=row_kl.device, dtype=torch.float32)
    loss = weighted_mean(row_kl, mask)
    train_rows = row_kl.reshape(-1)[mask.reshape(-1) > 0.0]
    if int(train_rows.numel()) == 0:
        train_rows = row_kl.reshape(-1)
    metrics = {
        "policy_anchor_loss": float(loss.detach().item()),
        "policy_anchor_kl_mean": float(train_rows.detach().mean().item()),
        "policy_anchor_kl_p95": float(torch.quantile(train_rows.detach(), 0.95).item()),
        "policy_anchor_candidate_count": float(current_log_probs.numel()),
    }
    return loss, metrics


def packed_candidate_anchor_top_action_loss(
    *,
    current_log_probs: Tensor,
    anchor_log_probs: Tensor,
    packed_offsets: Tensor,
    row_shape: tuple[int, int],
    loss_mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Imitate the anchor policy's top legal action on each row."""

    if current_log_probs.shape != anchor_log_probs.shape:
        raise ValueError(
            "current and anchor packed log-prob tensors must have identical shapes, "
            f"got {tuple(current_log_probs.shape)} and {tuple(anchor_log_probs.shape)}"
        )
    offsets = packed_offsets.to(device=current_log_probs.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    expected_rows = int(row_shape[0] * row_shape[1])
    if row_count != expected_rows:
        raise ValueError(f"packed row count {row_count} does not match row_shape {row_shape}")
    if tuple(loss_mask.shape) != tuple(row_shape):
        raise ValueError(f"loss_mask must match row_shape {row_shape}, got {tuple(loss_mask.shape)}")

    lengths = offsets[1:] - offsets[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, dtype=torch.long, device=current_log_probs.device),
        lengths,
    )
    if int(row_indices.numel()) != int(current_log_probs.numel()):
        raise ValueError("packed offsets do not match packed log-prob tensor length")
    current_logp = current_log_probs.to(dtype=torch.float32)
    anchor_logp = anchor_log_probs.to(device=current_logp.device, dtype=torch.float32)
    anchor_top = segment_max(anchor_logp, row_indices, row_count)
    current_top = segment_max(current_logp, row_indices, row_count)
    anchor_top_mask = anchor_logp == anchor_top.index_select(0, row_indices)
    current_top_mask = current_logp == current_top.index_select(0, row_indices)
    top_counts = torch.zeros((row_count,), dtype=torch.float32, device=current_logp.device)
    top_counts.scatter_add_(0, row_indices, anchor_top_mask.to(dtype=torch.float32))
    candidate_loss = torch.where(anchor_top_mask, -current_logp, torch.zeros_like(current_logp))
    row_loss = torch.zeros((row_count,), dtype=torch.float32, device=current_logp.device)
    row_loss.scatter_add_(0, row_indices, candidate_loss)
    row_loss = row_loss / torch.clamp(top_counts, min=1.0)
    agreement_candidates = (anchor_top_mask & current_top_mask).to(dtype=torch.float32)
    row_agreement = torch.zeros((row_count,), dtype=torch.float32, device=current_logp.device)
    row_agreement.scatter_add_(0, row_indices, agreement_candidates)
    row_agreement = torch.clamp(row_agreement, max=1.0)
    row_loss = row_loss.reshape(row_shape)
    row_agreement = row_agreement.reshape(row_shape)
    mask = loss_mask.to(device=row_loss.device, dtype=torch.float32)
    loss = weighted_mean(row_loss, mask)
    agreement = weighted_mean(row_agreement, mask)
    train_rows = row_loss.reshape(-1)[mask.reshape(-1) > 0.0]
    if int(train_rows.numel()) == 0:
        train_rows = row_loss.reshape(-1)
    metrics = {
        "policy_anchor_top_action_loss": float(loss.detach().item()),
        "policy_anchor_top_action_agreement": float(agreement.detach().item()),
        "policy_anchor_top_action_loss_p95": float(torch.quantile(train_rows.detach(), 0.95).item()),
    }
    return loss, metrics
