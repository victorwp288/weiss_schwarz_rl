"""Lightweight masked PPO baseline learner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from weiss_rl.diagnostics.training_logger import TrainingMetrics

from .impala import (
    ImpalaLearner,
    _batch_value,
    _masked_action_logp_and_entropy,
    _packed_action_logp_and_entropy,
)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denominator


def _masked_advantage_normalize(advantages: Tensor, mask: Tensor) -> Tensor:
    active = mask > 0
    if not bool(active.any().item()):
        return advantages
    active_values = advantages[active]
    centered = advantages - active_values.mean()
    std = active_values.std(unbiased=False)
    if bool((std > 1e-6).item()):
        centered = centered / std
    return centered


def _explained_variance(targets: Tensor, predictions: Tensor, mask: Tensor) -> float:
    active = mask > 0
    if not bool(active.any().item()):
        return 0.0
    target_values = targets[active]
    prediction_values = predictions[active]
    target_var = torch.var(target_values, unbiased=False)
    if bool((target_var <= 1e-12).item()):
        return 0.0
    residual_var = torch.var(target_values - prediction_values, unbiased=False)
    return float((1.0 - (residual_var / target_var)).detach().cpu().item())


@dataclass(slots=True)
class PpoLiteLearner(ImpalaLearner):
    ppo_clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    target_kl: float = 0.0
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        super(PpoLiteLearner, self).__post_init__()
        if float(self.trajectory_retention_coef) != 0.0:
            raise ValueError("trajectory_retention_coef is only supported by IMPALA/V-trace")

    def update(self, batch: Any) -> dict[str, float]:
        self.update_count += 1
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size

        elapsed = time.time() - self.start_time
        throughput_samples_per_sec = self.total_samples_processed / max(elapsed, 1e-6)
        throughput_updates_per_sec = self.update_count / max(elapsed, 1e-6)

        if self.checkpoint_dir and self.update_count % self.checkpoint_interval_updates == 0:
            self.policy_version += 1
            self._write_checkpoint_metadata()

        metrics: dict[str, float] = {
            "loss": 0.0,
            "throughput_samples_per_sec": throughput_samples_per_sec,
            "throughput_updates_per_sec": throughput_updates_per_sec,
            "entropy_coef": float(self.entropy_coef),
        }
        has_training_inputs = _batch_value(batch, "obs") is not None
        if not has_training_inputs:
            return metrics

        missing = [
            key
            for key in ("obs", "actions", "advantages", "returns", "old_logp", "old_values")
            if _batch_value(batch, key) is None
        ]
        if not self._has_legal_actions(batch):
            missing.append("legal_actions")
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                "batch must include obs, actions, legality, returns, advantages, old_logp, and old_values; "
                f"missing {missing_fields}"
            )
        if self.model is None:
            raise ValueError("PpoLiteLearner requires a model to run an optimizer step")

        optimizer = self._optimizer_for_step()
        epoch_metrics: list[dict[str, float]] = []
        last_context: dict[str, Any] = {}
        completed_epochs = 0
        for _epoch_index in range(int(self.ppo_epochs)):
            self.model.train()
            loss, loss_metrics, loss_context = self._loss_and_metrics_with_context(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self._ensure_finite_gradients(batch=batch, context=loss_context, grad_norm=grad_norm)
            optimizer.step()

            completed_epochs += 1
            last_context = loss_context
            epoch_metric = dict(loss_metrics)
            epoch_metric["grad_norm"] = float(grad_norm)
            epoch_metrics.append(epoch_metric)
            if self.target_kl > 0.0 and epoch_metric.get("approx_kl", 0.0) >= self.target_kl:
                break

        averaged_metrics = _mean_metric_dicts(epoch_metrics)
        averaged_metrics["ppo_epochs_completed"] = float(completed_epochs)
        metrics.update(averaged_metrics)
        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(metrics, batch, context=last_context)
            self.last_log_time = time.time()
            self.last_log_update = self.update_count
        return metrics

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("PpoLiteLearner requires a model to compute losses")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        logits, values = self._forward_time_major(
            obs,
            initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
            to_play_seat=_batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            legal_actions=_batch_value(batch, "legal_actions"),
        )
        packed_legal = self._resolve_packed_legal_actions(batch, expected_shape=obs.shape[:2])
        legal_mask = (
            None
            if packed_legal is not None
            else self._resolve_legal_mask(
                batch,
                expected_shape=obs.shape[:2],
                action_dim=logits.shape[-1],
            )
        )
        context: dict[str, Any] = {
            "logits": logits.detach(),
            "values": values.detach(),
        }
        self._ensure_finite_tensor("forward_logits", logits, batch=batch, context=context)
        self._ensure_finite_tensor("forward_values", values, batch=batch, context=context)

        if packed_legal is not None:
            packed_ids, packed_offsets = packed_legal
            action_logp, entropy = _packed_action_logp_and_entropy(
                logits,
                packed_ids,
                packed_offsets,
                actions,
                pass_action_id=self.pass_action_id,
            )
        else:
            assert legal_mask is not None
            action_logp, entropy = _masked_action_logp_and_entropy(
                logits,
                legal_mask,
                actions,
                pass_action_id=self.pass_action_id,
            )
        context["action_logp"] = action_logp.detach()
        context["entropy"] = entropy.detach()
        self._ensure_finite_tensor("action_logp", action_logp, batch=batch, context=context)
        self._ensure_finite_tensor("entropy", entropy, batch=batch, context=context)

        returns = self._float_target(_batch_value(batch, "returns"), expected_shape=values.shape, like=values)
        advantages = self._float_target(_batch_value(batch, "advantages"), expected_shape=values.shape, like=values)
        old_logp = self._float_target(_batch_value(batch, "old_logp"), expected_shape=values.shape, like=values)
        old_values = self._float_target(_batch_value(batch, "old_values"), expected_shape=values.shape, like=values)
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        if self.normalize_advantages:
            advantages = _masked_advantage_normalize(advantages, loss_mask)

        context["returns"] = returns.detach()
        context["advantages"] = advantages.detach()
        context["old_logp"] = old_logp.detach()
        context["old_values"] = old_values.detach()
        context["policy_train_mask"] = loss_mask.detach()

        log_ratio = action_logp - old_logp
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - float(self.ppo_clip_epsilon),
            max=1.0 + float(self.ppo_clip_epsilon),
        )
        surrogate_unclipped = ratio * advantages
        surrogate_clipped = clipped_ratio * advantages
        policy_loss = -_masked_mean(torch.minimum(surrogate_unclipped, surrogate_clipped), loss_mask)

        if self.value_clip_epsilon > 0.0:
            clipped_values = old_values + torch.clamp(
                values - old_values,
                min=-float(self.value_clip_epsilon),
                max=float(self.value_clip_epsilon),
            )
            value_loss_raw = (values - returns) ** 2
            value_loss_clipped = (clipped_values - returns) ** 2
            value_loss = _masked_mean(torch.maximum(value_loss_raw, value_loss_clipped), loss_mask)
        else:
            value_loss = _masked_mean((values - returns) ** 2, loss_mask)
        entropy_mean = _masked_mean(entropy, loss_mask)
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_mean)

        context["policy_loss"] = policy_loss.detach()
        context["value_loss"] = value_loss.detach()
        context["entropy_mean"] = entropy_mean.detach()
        context["total_loss"] = total_loss.detach()
        self._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
        self._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
        self._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
        self._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)

        clip_mask = (torch.abs(ratio - 1.0) > float(self.ppo_clip_epsilon)).to(dtype=values.dtype)
        approx_kl = _masked_mean(old_logp - action_logp, loss_mask)
        clip_fraction = _masked_mean(clip_mask, loss_mask)
        explained_variance = _explained_variance(returns, values, loss_mask)
        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "policy_train_fraction": float(loss_mask.mean().detach()),
            "approx_kl": float(approx_kl.detach()),
            "clip_fraction": float(clip_fraction.detach()),
            "explained_variance": explained_variance,
        }
        return total_loss, metrics, context

    def _log_metrics(
        self, update_metrics: dict[str, float], batch: Any, *, context: dict[str, Any] | None = None
    ) -> None:
        if not self.logger:
            return
        if context is None:
            context = {}
        elapsed = time.time() - self.start_time
        metrics = TrainingMetrics(
            update_count=self.update_count,
            wall_clock_seconds=elapsed,
            wall_clock_ms=int(elapsed * 1000),
            policy_version=self.policy_version,
            loss=float(update_metrics.get("loss", 0.0)),
            throughput_samples_per_sec=float(update_metrics.get("throughput_samples_per_sec", 0.0)),
            throughput_updates_per_sec=float(update_metrics.get("throughput_updates_per_sec", 0.0)),
            vtrace_rho_mean=0.0,
            vtrace_rho_p50=0.0,
            vtrace_rho_p90=0.0,
            vtrace_rho_p99=0.0,
            vtrace_clip_rate=0.0,
            vtrace_c_clipped_rate=0.0,
            kl_divergence=float(update_metrics.get("approx_kl", 0.0)),
            value_loss=float(update_metrics.get("value_loss", 0.0)),
            actor_loss=float(update_metrics.get("policy_loss", 0.0)),
            entropy=float(update_metrics.get("entropy", 0.0)),
            custom_metrics={
                "clip_fraction": float(update_metrics.get("clip_fraction", 0.0)),
                "explained_variance": float(update_metrics.get("explained_variance", 0.0)),
                "ppo_epochs_completed": float(update_metrics.get("ppo_epochs_completed", 0.0)),
                "policy_train_fraction": float(update_metrics.get("policy_train_fraction", 0.0)),
            },
        )
        self.logger.log(metrics)


def _mean_metric_dicts(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    return {key: float(np.mean([float(record[key]) for record in records if key in record])) for key in keys}
