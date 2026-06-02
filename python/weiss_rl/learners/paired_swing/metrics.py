"""Metric and output assembly for paired-swing replay losses."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from weiss_rl.learners.tensor_ops import weighted_mean


@dataclass(frozen=True)
class PairedSwingSupportedRows:
    margins: Tensor
    positive_metric_logp: Tensor
    negative_metric_logp: Tensor
    supported_weight: Tensor
    supported_weight_total: float
    metrics: dict[str, float]

    @property
    def has_weight(self) -> bool:
        return self.supported_weight_total > 0.0


def paired_swing_no_active_metrics(
    *,
    candidate_metrics: dict[str, float],
    metric_prefix: str,
) -> dict[str, float]:
    metrics = dict(candidate_metrics)
    metrics.update(
        {
            f"{metric_prefix}_rows": 0.0,
            f"{metric_prefix}_supported_fraction": 0.0,
            f"{metric_prefix}_loss": 0.0,
        }
    )
    return metrics


def paired_swing_supported_rows(
    *,
    margin_by_row: Tensor,
    positive_logp_by_row: Tensor,
    negative_logp_by_row: Tensor,
    supported: Tensor,
    flat_loss_mask: Tensor,
    raw_weight_total: float,
    packed_logits: Tensor,
    candidate_metrics: dict[str, float],
    metric_prefix: str,
) -> PairedSwingSupportedRows:
    supported_weight = flat_loss_mask[supported]
    supported_weight_total = float(supported_weight.sum().item()) if bool(supported.any().item()) else 0.0
    metrics = dict(candidate_metrics)
    metrics[f"{metric_prefix}_supported_fraction"] = supported_weight_total / max(raw_weight_total, 1.0e-8)
    metrics[f"{metric_prefix}_rows"] = float(supported.sum().item())
    return PairedSwingSupportedRows(
        margins=margin_by_row[supported].to(dtype=packed_logits.dtype),
        positive_metric_logp=positive_logp_by_row[supported],
        negative_metric_logp=negative_logp_by_row[supported],
        supported_weight=supported_weight,
        supported_weight_total=supported_weight_total,
        metrics=metrics,
    )


def paired_swing_no_supported_metrics(
    *,
    row_metrics: dict[str, float],
    metric_prefix: str,
) -> dict[str, float]:
    metrics = dict(row_metrics)
    metrics[f"{metric_prefix}_loss"] = 0.0
    return metrics


def paired_swing_final_metrics(
    *,
    row_metrics: dict[str, float],
    loss: Tensor,
    margin_mean: Tensor,
    satisfied_fraction: Tensor,
    normalized_scope: str,
    normalized_compare_to: str,
    margin_retention_coef: float,
    margin_retention_margin: float,
    top_action_retention_coef: float,
    top_action_retention_margin: float,
    positive_metric_logp: Tensor,
    negative_metric_logp: Tensor,
    supported_weight: Tensor,
    scope_metrics: dict[str, float],
    retention_metrics: dict[str, float],
    top_retention_metrics: dict[str, float],
    metric_prefix: str,
) -> dict[str, float]:
    metrics = dict(row_metrics)
    metrics.update(
        {
            f"{metric_prefix}_loss": float(loss.detach().item()),
            f"{metric_prefix}_margin_mean": float(margin_mean.detach().item()),
            f"{metric_prefix}_satisfied_fraction": float(satisfied_fraction.detach().item()),
            f"{metric_prefix}_loss_scope_episode_mean": 1.0 if normalized_scope == "episode_mean" else 0.0,
            f"{metric_prefix}_loss_scope_label_mean": 1.0 if normalized_scope == "label_mean" else 0.0,
            f"{metric_prefix}_compare_to_top_other": 1.0 if normalized_compare_to == "top_other" else 0.0,
            f"{metric_prefix}_margin_retention_coef": float(margin_retention_coef),
            f"{metric_prefix}_margin_retention_margin": float(margin_retention_margin),
            f"{metric_prefix}_top_action_retention_coef": float(top_action_retention_coef),
            f"{metric_prefix}_top_action_retention_margin": float(top_action_retention_margin),
            f"{metric_prefix}_positive_logp_mean": float(
                weighted_mean(positive_metric_logp, supported_weight).detach().item()
            ),
            f"{metric_prefix}_negative_logp_mean": float(
                weighted_mean(negative_metric_logp, supported_weight).detach().item()
            ),
            **scope_metrics,
        }
    )
    metrics.update(retention_metrics)
    metrics.update(top_retention_metrics)
    return metrics


def paired_swing_output_tensors(
    *,
    margins: Tensor,
    retention_tensors: dict[str, Tensor],
    top_retention_tensors: dict[str, Tensor],
) -> dict[str, Tensor]:
    tensors = {"paired_swing_margins": margins.detach()}
    tensors.update(retention_tensors)
    tensors.update(top_retention_tensors)
    return tensors


__all__ = [
    "PairedSwingSupportedRows",
    "paired_swing_final_metrics",
    "paired_swing_no_active_metrics",
    "paired_swing_no_supported_metrics",
    "paired_swing_output_tensors",
    "paired_swing_supported_rows",
]
