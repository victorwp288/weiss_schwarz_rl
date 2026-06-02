"""Context and metric assembly for IMPALA paired-swing replay."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from weiss_rl.learners.impala.paired_swing_candidates import PairedSwingCandidateView


def build_paired_swing_auxiliary_context(
    *,
    weighted_loss: Tensor,
    candidate_view: PairedSwingCandidateView,
    loss_mask: Tensor,
    swing_context: dict[str, Tensor],
) -> dict[str, Any]:
    return {
        "paired_swing_loss": weighted_loss.detach(),
        "logits": None if candidate_view.logits is None else candidate_view.logits.detach(),
        "packed_logits": candidate_view.packed_view.logits.detach(),
        "values": None if candidate_view.values is None else candidate_view.values.detach(),
        "policy_train_mask": loss_mask.detach(),
        **swing_context,
    }


def build_paired_swing_auxiliary_metrics(
    *,
    weighted_loss: Tensor,
    coef: float,
    margin: float,
    positive_action_source: str,
    negative_action_source: str,
    loss_scope: str,
    compare_to: str,
    margin_retention_coef: float,
    margin_retention_margin: float,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    swing_metrics: dict[str, float],
) -> dict[str, float]:
    metrics = {
        "loss": float(weighted_loss.detach().item()),
        "paired_swing_weighted_loss": float(weighted_loss.detach().item()),
        "paired_swing_coef": float(coef),
        "paired_swing_margin": float(margin),
        "paired_swing_margin_retention_coef": float(margin_retention_coef),
        "paired_swing_margin_retention_margin": float(margin_retention_margin),
        "paired_swing_top_action_retention_coef": float(top_action_retention_coef),
        "paired_swing_top_action_retention_margin": float(top_action_retention_margin),
        "paired_swing_positive_action_source_teacher": 1.0 if positive_action_source == "teacher_action" else 0.0,
        "paired_swing_negative_action_source_teacher": 1.0 if negative_action_source == "teacher_action" else 0.0,
        "paired_swing_loss_scope_episode_mean": 1.0 if str(loss_scope) == "episode_mean" else 0.0,
        "paired_swing_loss_scope_label_mean": 1.0 if str(loss_scope) == "label_mean" else 0.0,
        "paired_swing_compare_to_top_other": 1.0 if str(compare_to).strip().lower() == "top_other" else 0.0,
    }
    metrics.update(swing_metrics)
    return metrics


__all__ = ["build_paired_swing_auxiliary_context", "build_paired_swing_auxiliary_metrics"]
