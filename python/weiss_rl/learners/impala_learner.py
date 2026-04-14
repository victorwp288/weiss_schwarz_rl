"""IMPALA learner helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.learners.vtrace import VTraceTargets, VtraceMetrics, compute_vtrace_metrics
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.replay.bundles import write_fault_bundle
from weiss_rl.training_logger import TrainingLogger, TrainingMetrics


VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)
_MAX_LOG_RHO_TORCH = float(np.log(np.finfo(np.float32).max))


def _nonfinite_indices(values: Tensor | np.ndarray) -> np.ndarray:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return np.argwhere(~np.isfinite(array)).astype(np.int64, copy=False)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    flat_rhos = np.asarray(result.rhos, dtype=np.float64).reshape(-1)
    if flat_rhos.size == 0:
        raise ValueError("result.rhos must not be empty")

    metrics = {
        f"vtrace_rho_p{percentile}": float(np.percentile(flat_rhos, percentile))
        for percentile in VTRACE_RHO_PERCENTILES
    }
    metrics["vtrace_rho_clip_rate"] = float(np.mean(flat_rhos > rho_bar))
    metrics["vtrace_c_clip_rate"] = float(np.mean(flat_rhos > c_bar))
    return metrics


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _compute_vtrace_targets_torch(
    rewards: Tensor,
    values: Tensor,
    discounts: Tensor,
    behavior_logp: Tensor,
    target_logp: Tensor,
    *,
    rho_bar: float,
    c_bar: float,
) -> tuple[Tensor, Tensor, Tensor]:
    log_rhos = torch.clamp(target_logp - behavior_logp, max=_MAX_LOG_RHO_TORCH)
    rhos = torch.exp(log_rhos).clamp(max=torch.finfo(torch.float32).max)
    rho_bar_tensor = torch.full((), float(rho_bar), dtype=rhos.dtype, device=rhos.device)
    c_bar_tensor = torch.full((), float(c_bar), dtype=rhos.dtype, device=rhos.device)
    clipped_rhos = torch.minimum(rho_bar_tensor, rhos)
    clipped_cs = torch.minimum(c_bar_tensor, rhos)

    acc = torch.zeros_like(values[-1])
    vs_minus_v_xs = torch.zeros_like(rewards)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = clipped_rhos[t] * (rewards[t] + discounts[t] * values[t + 1] - values[t])
        acc = delta + discounts[t] * clipped_cs[t] * acc
        vs_minus_v_xs[t] = acc

    vs = values[:-1] + vs_minus_v_xs
    next_vs = torch.cat((vs[1:], values[-1:].clone()), dim=0)
    pg_advantages = clipped_rhos * (rewards + discounts * next_vs - values[:-1])
    return vs, pg_advantages, rhos


def _masked_log_probs_and_entropy(logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")

    mask = legal_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_legal = mask.any(dim=1, keepdim=True)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    row_max = torch.where(has_legal, row_max, torch.zeros_like(row_max))

    shifted = torch.where(mask, logits - row_max, torch.full_like(logits, float("-inf")))
    exp_shifted = torch.where(mask, torch.exp(shifted), torch.zeros_like(logits))
    denom = exp_shifted.sum(dim=1, keepdim=True)
    safe_denom = torch.where(has_legal, denom, torch.ones_like(denom))
    log_probs = torch.where(mask, shifted - torch.log(safe_denom), torch.full_like(logits, float("-inf")))

    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=1)
    return log_probs, entropy


def _masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_mask = legal_mask.reshape(-1, logits.shape[-1]).to(dtype=torch.bool)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]

    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")

    empty_rows = ~flat_mask.any(dim=1)
    row_actions = flat_actions.unsqueeze(1)
    action_is_legal = flat_mask.gather(dim=1, index=row_actions).squeeze(1)
    illegal_rows = (~empty_rows) & (~action_is_legal)
    if bool(illegal_rows.any().item()):
        row_index = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        action = int(flat_actions[row_index].item())
        raise ValueError(f"illegal action {action} for row {row_index}")

    log_probs, entropy = _masked_log_probs_and_entropy(flat_logits, flat_mask)
    selected_logp = log_probs.gather(dim=1, index=row_actions).squeeze(1)

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )
        selected_logp = torch.where(empty_rows, torch.zeros_like(selected_logp), selected_logp)
        entropy = torch.where(empty_rows, torch.zeros_like(entropy), entropy)

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


def _packed_action_logp_and_entropy(
    logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]
    row_count = flat_logits.shape[0]

    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()):
        raise ValueError("legal_offsets must end at len(legal_ids)")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")
    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")
    if bool((ids < 0).any().item()) or bool((ids >= action_space).any().item()):
        raise ValueError(f"packed legal ids must be in [0, {action_space})")

    selected_logp = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    entropy = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        row_ids = torch.repeat_interleave(non_empty_rows, non_empty_widths)
        legal_logits = flat_logits[row_ids, ids]
        segment_max = torch.segment_reduce(legal_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max, non_empty_widths)
        shifted = legal_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)
        entropy_terms = -(torch.exp(log_probs) * log_probs)
        entropy_non_empty = torch.segment_reduce(entropy_terms, reduce="sum", lengths=non_empty_widths)
        entropy[non_empty_rows] = entropy_non_empty

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(matches.to(dtype=flat_logits.dtype), reduce="sum", lengths=non_empty_widths)
        illegal_rows = match_counts != 1.0
        if bool(illegal_rows.any().item()):
            bad_row = int(non_empty_rows[torch.nonzero(illegal_rows, as_tuple=False)[0].item()].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        selected_logp[non_empty_rows] = selected_non_empty

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


@dataclass(slots=True)
class ImpalaLearner:
    model: nn.Module | None = None
    compiled_model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_norm_clip: float = 40.0
    mixed_precision: bool = False
    checkpoint_dir: Path | None = None
    fault_dir: Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: Path | None = None
    logging_interval_updates: int = 100
    vtrace_rho_bar: float = 2.4
    vtrace_c_bar: float = 1.0
    pass_action_id: int | None = None

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)
    _amp_enabled: bool = field(default=False, init=False)
    _amp_device_type: str = field(default="cpu", init=False)
    _grad_scaler: torch.amp.GradScaler | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)
        self._refresh_acceleration_state()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def _refresh_acceleration_state(self) -> None:
        if self.model is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        self._amp_device_type = parameter.device.type
        self._amp_enabled = bool(self.mixed_precision and self._amp_device_type == "cuda")
        self._grad_scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self._amp_enabled
            else None
        )

    def update(self, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
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
        vtrace_result = _batch_value(batch, "vtrace_result")

        has_training_inputs = _batch_value(batch, "obs") is not None
        if has_training_inputs:
            missing = [key for key in ("obs", "actions") if _batch_value(batch, key) is None]
            if not self._has_legal_actions(batch):
                missing.append("legal_actions")
            has_vtrace_targets = isinstance(_batch_value(batch, "vtrace_result"), VTraceTargets)
            has_raw_vtrace_inputs = all(
                _batch_value(batch, key) is not None
                for key in ("rewards", "discounts", "behavior_logp", "behavior_values", "bootstrap_value")
            )
            if not has_vtrace_targets and not has_raw_vtrace_inputs:
                missing.append("vtrace_result_or_raw_inputs")
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(
                    "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs for learner updates; "
                    f"missing {missing_fields}"
                )
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model to run an optimizer step")

            self.model.train()
            if self.compiled_model is not None:
                self.compiled_model.train()
            with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
                loss, loss_metrics, loss_context = self._loss_and_metrics_with_context(batch)
            optimizer = self._optimizer_for_step()
            optimizer.zero_grad(set_to_none=True)
            loss_scale_before = None
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            if self._grad_scaler is not None:
                bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
                gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
                if gradients_finite:
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if loss_scale_before is not None:
                        try:
                            self._grad_scaler.update(loss_scale_before * 0.5)
                        except TypeError:
                            self._grad_scaler.update()
                    else:
                        self._grad_scaler.update()
                loss_scale_after = float(self._grad_scaler.get_scale())
                gradient_overflow = (not gradients_finite) or bool(
                    loss_scale_before is not None and loss_scale_after < loss_scale_before
                )
                if gradient_overflow:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 1.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
                else:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 0.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
            else:
                self._ensure_finite_gradients(batch=batch, context=loss_context, grad_norm=grad_norm)
                optimizer.step()
                metrics.update(loss_metrics)
                metrics["grad_norm"] = float(grad_norm)

        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
            c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))

        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(metrics, batch)
            self.last_log_time = time.time()
            self.last_log_update = self.update_count

        return metrics

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        vtrace_result = _batch_value(batch, "vtrace_result")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        logits, values = self._forward_time_major(
            obs,
            initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
            to_play_seat=_batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
        )
        packed_legal = self._resolve_packed_legal_actions(batch, expected_shape=obs.shape[:2])
        legal_mask = None
        if packed_legal is None:
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")

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

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
        c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
        if isinstance(vtrace_result, VTraceTargets):
            targets = self._float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
            advantages = self._float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)
            rhos_for_metrics = self._float_target(vtrace_result.rhos, expected_shape=values.shape, like=values)
            raw_rewards = _batch_value(batch, "rewards")
            if raw_rewards is None:
                rewards_for_metrics = torch.zeros_like(values)
            else:
                rewards_for_metrics = self._float_target(raw_rewards, expected_shape=values.shape, like=values)
        else:
            rewards = self._float_target(_batch_value(batch, "rewards"), expected_shape=values.shape, like=values)
            discounts = self._float_target(_batch_value(batch, "discounts"), expected_shape=values.shape, like=values)
            behavior_logp = self._float_target(_batch_value(batch, "behavior_logp"), expected_shape=values.shape, like=values)
            behavior_values = self._float_target(_batch_value(batch, "behavior_values"), expected_shape=values.shape, like=values)
            bootstrap_value = self._float_input(_batch_value(batch, "bootstrap_value"))
            if bootstrap_value.ndim != 1 or bootstrap_value.shape[0] != values.shape[1]:
                raise ValueError(f"bootstrap_value must have shape ({values.shape[1]},), got {tuple(bootstrap_value.shape)}")
            full_values = torch.cat([behavior_values, bootstrap_value.unsqueeze(0)], dim=0)
            # Use the current learner policy log-prob for the V-trace target policy.
            # Passing behavior_logp twice silently forces rho=1 and disables off-policy correction.
            targets, advantages, rhos_for_metrics = _compute_vtrace_targets_torch(
                rewards,
                full_values,
                discounts,
                behavior_logp,
                action_logp,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
            rewards_for_metrics = rewards
        context["targets"] = targets.detach()
        context["advantages"] = advantages.detach()
        context["vtrace_rhos"] = rhos_for_metrics.detach()
        context["rewards"] = rewards_for_metrics.detach()
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        context["policy_train_mask"] = loss_mask.detach()
        loss_denominator = torch.clamp(loss_mask.sum(), min=1.0)

        policy_loss = -((action_logp * advantages) * loss_mask).sum() / loss_denominator
        value_loss = (((values - targets) ** 2) * loss_mask).sum() / loss_denominator
        entropy_mean = (entropy * loss_mask).sum() / loss_denominator
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_mean)

        context["policy_loss"] = policy_loss.detach()
        context["value_loss"] = value_loss.detach()
        context["entropy_mean"] = entropy_mean.detach()
        context["total_loss"] = total_loss.detach()
        self._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
        self._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
        self._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
        self._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)

        rho_metrics = rhos_for_metrics.detach().reshape(-1).to(dtype=torch.float32)
        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "policy_train_fraction": float(loss_mask.mean().detach()),
            "reward_mean": float(rewards_for_metrics.detach().mean().item()),
            "reward_abs_mean": float(rewards_for_metrics.detach().abs().mean().item()),
            "reward_nonzero_fraction": float((rewards_for_metrics.detach() != 0).float().mean().item()),
            "advantage_mean": float(advantages.detach().mean().item()),
            "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
            "target_mean": float(targets.detach().mean().item()),
            "target_abs_mean": float(targets.detach().abs().mean().item()),
            "vtrace_rho_p50": float(torch.quantile(rho_metrics, 0.50).item()),
            "vtrace_rho_p90": float(torch.quantile(rho_metrics, 0.90).item()),
            "vtrace_rho_p95": float(torch.quantile(rho_metrics, 0.95).item()),
            "vtrace_rho_p99": float(torch.quantile(rho_metrics, 0.99).item()),
            "vtrace_rho_clip_rate": float((rhos_for_metrics.detach() > rho_bar).float().mean().item()),
            "vtrace_c_clip_rate": float((rhos_for_metrics.detach() > c_bar).float().mean().item()),
        }
        return total_loss, metrics, context

    def _optimizer_for_step(self) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def _forward_time_major(
        self,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
    ) -> tuple[Tensor, Tensor]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        acting_seat = self._prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
        )
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []

        if acting_seat is None:
            hidden_state = self._prepare_legacy_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
            for step_obs in obs.unbind(dim=0):
                step_logits, step_value, hidden_state = forward_model(step_obs, hidden_state)
                logits_steps.append(torch.as_tensor(step_logits))
                value_steps.append(torch.as_tensor(step_value))
                hidden_state = torch.as_tensor(hidden_state)
            return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0)

        seat_hidden_state = self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
        for step_obs, step_seat in zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True):
            step_logits, step_value, seat_hidden_state = forward_model.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden_state,
            )
            logits_steps.append(torch.as_tensor(step_logits))
            value_steps.append(torch.as_tensor(step_value))
            seat_hidden_state = torch.as_tensor(seat_hidden_state)

        return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0)

    def _require_obs(self, value: Any) -> Tensor:
        tensor = self._float_input(value)
        if tensor.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(tensor.shape)}")
        return tensor

    def _require_actions(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._long_input(value)
        if tensor.shape != expected_shape:
            raise ValueError(f"actions must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _require_legal_mask(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._bool_input(value)
        if tensor.ndim != 3 or tensor.shape[:2] != expected_shape:
            expected = (int(expected_shape[0]), int(expected_shape[1]), "action")
            raise ValueError(f"legal_mask must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _has_legal_actions(self, batch: Any) -> bool:
        if _batch_value(batch, "legal_actions") is not None:
            return True
        if _batch_value(batch, "legal_mask") is not None:
            return True
        return _batch_value(batch, "legal_ids") is not None and _batch_value(batch, "legal_offsets") is not None

    def _resolve_legal_mask(self, batch: Any, *, expected_shape: torch.Size, action_dim: int) -> Tensor:
        legal_actions = _batch_value(batch, "legal_actions")
        if isinstance(legal_actions, LegalActionBatch):
            mask = legal_actions.to_mask(
                expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
                action_space=action_dim,
            )
            return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

        legal_mask = _batch_value(batch, "legal_mask")
        if legal_mask is not None:
            return self._require_legal_mask(legal_mask, expected_shape=expected_shape)

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            raise ValueError("batch must include either legal_actions, legal_mask, or legal_ids/legal_offsets")
        mask = LegalActionBatch.from_packed(legal_ids, legal_offsets).to_mask(
            expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
            action_space=action_dim,
        )
        return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

    def _resolve_packed_legal_actions(self, batch: Any, *, expected_shape: torch.Size) -> tuple[Tensor, Tensor] | None:
        legal_actions = _batch_value(batch, "legal_actions")
        if isinstance(legal_actions, LegalActionBatch) and legal_actions.ids is not None and legal_actions.offsets is not None:
            ids = torch.as_tensor(legal_actions.ids, device=self._model_parameter().device, dtype=torch.long)
            offsets = torch.as_tensor(legal_actions.offsets, device=self._model_parameter().device, dtype=torch.long)
            expected_rows = int(expected_shape[0] * expected_shape[1])
            if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
                raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
            return ids, offsets

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            return None
        ids = torch.as_tensor(legal_ids, device=self._model_parameter().device, dtype=torch.long)
        offsets = torch.as_tensor(legal_offsets, device=self._model_parameter().device, dtype=torch.long)
        expected_rows = int(expected_shape[0] * expected_shape[1])
        if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
            raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
        return ids, offsets

    def _float_target(self, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"target must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _prepare_legacy_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                "initial_hidden_state must be 2D (batch, hidden_size) when to_play_seat/actor is absent, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        return tensor

    def _prepare_seat_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 3:
            raise ValueError(
                "initial_hidden_state must be 3D (batch, seat, hidden_size) when to_play_seat/actor is present, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        if tensor.shape[1] != 2:
            raise ValueError(f"initial_hidden_state seat mismatch: expected 2, got {tensor.shape[1]}")
        return tensor

    def _prepare_acting_seat_batch(
        self,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        seat_tensor = self._optional_time_major_seat_field(
            to_play_seat,
            field_name="to_play_seat",
            expected_shape=expected_shape,
        )
        actor_tensor = self._optional_time_major_seat_field(
            actor,
            field_name="actor",
            expected_shape=expected_shape,
        )

        if seat_tensor is None:
            return actor_tensor
        if actor_tensor is None:
            return seat_tensor
        if not torch.equal(seat_tensor, actor_tensor):
            raise ValueError("actor must match to_play_seat when both are provided")
        return seat_tensor

    def _optional_time_major_seat_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{field_name} values must be 0 or 1")
        return tensor

    def _optional_time_major_loss_mask(
        self,
        value: Any,
        *,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"policy_train_mask must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor.clamp(min=0.0, max=1.0)

    def _float_input(self, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self, value: Any, *, dtype: torch.dtype) -> Tensor:
        if value is None:
            raise ValueError("batch field is required")
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        return tensor.to(dtype=dtype)

    def _model_parameter(self) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter

    def _batch_size(self, batch: Any) -> int:
        for key in ("rewards", "actions", "logits", "obs"):
            value = _batch_value(batch, key)
            if value is not None:
                return int(np.asarray(value).size)
        return 1

    def _fault_dir_path(self) -> Path:
        if self.fault_dir is not None:
            return self.fault_dir
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir / "faults"
        if self.logs_dir is not None:
            return self.logs_dir / "faults"
        return Path("faults")

    def _batch_fault_snapshot(self, batch: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key in (
            "obs",
            "actions",
            "legal_mask",
            "to_play_seat",
            "actor",
            "initial_hidden_state",
            "vtrace_rho_bar",
            "vtrace_c_bar",
        ):
            value = _batch_value(batch, key)
            if value is not None:
                snapshot[key] = value
        vtrace_result = _batch_value(batch, "vtrace_result")
        if vtrace_result is not None:
            snapshot["vtrace_result"] = vtrace_result
        return snapshot

    def _write_numeric_fault_bundle(self, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return write_fault_bundle(
            fault_dir=self._fault_dir_path(),
            prefix="learner_numeric_fault",
            payload={
                "format": "numeric_fault_bundle",
                "component": "impala_learner",
                "stage": stage,
                "update_count": self.update_count,
                "policy_version": self.policy_version,
                "batch_size": self._batch_size(batch),
                "pass_action_id": self.pass_action_id,
                "batch": self._batch_fault_snapshot(batch),
                "context": context,
            },
        )

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        if bool(torch.isfinite(tensor).all().item()):
            return
        fault_context = dict(context)
        fault_context[name] = tensor.detach()
        fault_context[f"{name}_nonfinite_indices"] = _nonfinite_indices(tensor)
        fault_path = self._write_numeric_fault_bundle(stage=name, batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner {name}; wrote fault bundle to {fault_path}")

    def _collect_nonfinite_gradients(self, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        model = self.model
        if model is None:
            raise ValueError("ImpalaLearner requires a model")

        bad_gradients = {
            name: parameter.grad.detach()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        }
        grad_norm_tensor = torch.as_tensor(grad_norm)
        return bad_gradients, grad_norm_tensor

    def _ensure_finite_gradients(self, *, batch: Any, context: dict[str, Any], grad_norm: Tensor) -> None:
        bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
        if not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item()):
            return

        self._raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
        )

    def _raise_for_nonfinite_gradients(
        self,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:

        fault_context = dict(context)
        fault_context["grad_norm"] = grad_norm_tensor.detach()
        fault_context["grad_norm_nonfinite_indices"] = _nonfinite_indices(grad_norm_tensor)
        if bad_gradients:
            fault_context["bad_gradient_names"] = sorted(bad_gradients)
            fault_context["bad_gradients"] = bad_gradients
        fault_path = self._write_numeric_fault_bundle(stage="gradients", batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner gradients; wrote fault bundle to {fault_path}")

    def _write_checkpoint_metadata(self) -> None:
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata_path = self.checkpoint_dir / f"checkpoint_metadata_{self.update_count}.json"
        checkpoint_metadata_path.write_text(
            json.dumps(
                {
                    "format": "checkpoint_metadata",
                    "parameters_included": False,
                    "update_count": self.update_count,
                    "policy_version": self.policy_version,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint metadata: {checkpoint_metadata_path}")

    def _log_metrics(self, update_metrics: dict[str, float], batch: Any) -> None:
        if not self.logger:
            return

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        vtrace_metrics = compute_vtrace_metrics(
            batch,
            rho_bar=self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
            c_bar=self.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = TrainingMetrics(
            update_count=self.update_count,
            wall_clock_seconds=elapsed,
            wall_clock_ms=int(elapsed * 1000),
            policy_version=self.policy_version,
            loss=float(update_metrics.get("loss", 0.0)),
            throughput_samples_per_sec=float(update_metrics.get("throughput_samples_per_sec", 0.0)),
            throughput_updates_per_sec=float(update_metrics.get("throughput_updates_per_sec", 0.0)),
            vtrace_rho_mean=vtrace_metrics.rho_mean,
            vtrace_rho_p50=float(update_metrics.get("vtrace_rho_p50", vtrace_metrics.rho_p50)),
            vtrace_rho_p90=float(update_metrics.get("vtrace_rho_p90", vtrace_metrics.rho_p90)),
            vtrace_rho_p99=float(update_metrics.get("vtrace_rho_p99", vtrace_metrics.rho_p99)),
            vtrace_clip_rate=float(update_metrics.get("vtrace_rho_clip_rate", vtrace_metrics.clip_rate)),
            vtrace_c_clipped_rate=float(update_metrics.get("vtrace_c_clip_rate", vtrace_metrics.c_clipped_rate)),
            kl_divergence=vtrace_metrics.kl_divergence,
            value_loss=float(update_metrics.get("value_loss", 0.0)),
            actor_loss=float(update_metrics.get("policy_loss", 0.0)),
            entropy=float(update_metrics.get("entropy", vtrace_metrics.entropy)),
            custom_metrics=self._custom_log_metrics(update_metrics, vtrace_metrics),
        )
        self.logger.log(metrics)

    def _custom_log_metrics(
        self,
        update_metrics: dict[str, float],
        vtrace_metrics: VtraceMetrics,
    ) -> dict[str, float]:
        custom_metrics: dict[str, float] = {
            "vtrace_batch_metrics_available": float(np.isfinite(vtrace_metrics.rho_mean)),
        }
        if "vtrace_rho_p95" in update_metrics:
            custom_metrics["vtrace_rho_p95"] = float(update_metrics["vtrace_rho_p95"])
        if np.isfinite(vtrace_metrics.entropy):
            custom_metrics["vtrace_entropy"] = float(vtrace_metrics.entropy)
        return custom_metrics

    def get_policy_version(self) -> int:
        return self.policy_version
