"""IMPALA learner loss-metric assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog


def chosen_action_outcome_metrics(
    *,
    actions: Tensor,
    loss_mask: Tensor,
    rewards: Tensor,
    advantages: Tensor,
    action_catalog: ActionCatalog | None,
    pass_action_id: int | None,
) -> dict[str, float]:
    train_mask = loss_mask.detach().reshape(-1).to(dtype=torch.bool)
    train_count = int(train_mask.sum().item())
    if train_count == 0:
        return {}

    flat_actions = actions.detach().reshape(-1).to(device=train_mask.device, dtype=torch.long)
    flat_rewards = rewards.detach().reshape(-1).to(device=train_mask.device, dtype=torch.float32)
    flat_advantages = advantages.detach().reshape(-1).to(device=train_mask.device, dtype=torch.float32)
    metrics: dict[str, float] = {}

    def selected_fraction(selected: Tensor) -> float:
        return float((selected & train_mask).to(dtype=torch.float32).sum().item() / float(train_count))

    def selected_mean(values: Tensor, selected: Tensor) -> float:
        selected_train = selected & train_mask
        if not bool(selected_train.any().item()):
            return 0.0
        return float(values[selected_train].mean().item())

    if pass_action_id is not None:
        pass_selected = flat_actions == int(pass_action_id)
        nonpass_selected = ~pass_selected
        metrics.update(
            {
                "chosen_pass_train_fraction": selected_fraction(pass_selected),
                "chosen_pass_train_reward_mean": selected_mean(flat_rewards, pass_selected),
                "chosen_pass_train_advantage_mean": selected_mean(flat_advantages, pass_selected),
                "chosen_nonpass_train_reward_mean": selected_mean(flat_rewards, nonpass_selected),
                "chosen_nonpass_train_advantage_mean": selected_mean(flat_advantages, nonpass_selected),
            }
        )

    if action_catalog is None:
        return metrics

    def family_selected(family_name: str) -> Tensor:
        selected = torch.zeros_like(flat_actions, dtype=torch.bool)
        for family in action_catalog.families:
            if family.name != family_name:
                continue
            selected |= (flat_actions >= int(family.base)) & (flat_actions < int(family.base + family.count))
        return selected

    for family_name, metric_stem in (
        ("mulligan_confirm", "chosen_mulligan_confirm"),
        ("mulligan_select", "chosen_mulligan_select"),
        ("main_play_character", "chosen_main_play_character"),
        ("main_move", "chosen_main_move"),
        ("attack", "chosen_attack"),
    ):
        selected = family_selected(family_name)
        metrics[f"{metric_stem}_train_fraction"] = selected_fraction(selected)
        metrics[f"{metric_stem}_train_reward_mean"] = selected_mean(flat_rewards, selected)
        metrics[f"{metric_stem}_train_advantage_mean"] = selected_mean(flat_advantages, selected)
    return metrics


def _float_or_zero(value: Any) -> float:
    return float(value or 0.0)


def build_impala_loss_metrics(
    *,
    total_loss: Tensor,
    policy_loss: Tensor,
    value_loss: Tensor,
    entropy_mean: Tensor,
    entropy_scope: str,
    loss_mask: Tensor,
    value_loss_mask: Tensor,
    actions: Tensor,
    action_logp: Tensor,
    behavior_logp_for_mask: Tensor | None,
    rewards_for_metrics: Tensor,
    advantages: Tensor,
    targets: Tensor,
    rhos_for_metrics: Tensor,
    rho_bar: float,
    c_bar: float,
    action_catalog: ActionCatalog | None,
    pass_action_id: int | None,
    terminal_outcome_backfill_count: Any,
    terminal_outcome_backfill_total_micros: Any,
    terminal_outcome_trace_backfill_count: Any,
    terminal_outcome_trace_backfill_total_micros: Any,
    trajectory_retention_metrics: dict[str, float],
    policy_anchor_metrics: dict[str, float],
    teacher_metrics: dict[str, float],
) -> dict[str, float]:
    rho_metrics = rhos_for_metrics.detach().reshape(-1).to(dtype=torch.float32)
    train_rho_mask = loss_mask.detach().reshape(-1).to(device=rho_metrics.device, dtype=torch.bool)
    train_rho_metrics = rho_metrics[train_rho_mask]
    if int(train_rho_metrics.numel()) == 0:
        train_rho_metrics = rho_metrics

    logp_delta_metrics: dict[str, float] = {}
    if behavior_logp_for_mask is not None:
        logp_delta = (
            action_logp.detach().to(dtype=torch.float32)
            - behavior_logp_for_mask.detach().to(device=action_logp.device, dtype=torch.float32)
        ).reshape(-1)
        logp_delta_abs = logp_delta.abs()
        train_logp_delta_abs = logp_delta_abs[train_rho_mask]
        if int(train_logp_delta_abs.numel()) == 0:
            train_logp_delta_abs = logp_delta_abs
        logp_delta_metrics = {
            "target_behavior_logp_delta_mean": float(logp_delta.mean().item()),
            "target_behavior_logp_delta_abs_mean": float(logp_delta_abs.mean().item()),
            "target_behavior_logp_delta_abs_p95": float(torch.quantile(logp_delta_abs, 0.95).item()),
            "target_behavior_logp_delta_abs_p99": float(torch.quantile(logp_delta_abs, 0.99).item()),
            "target_behavior_train_logp_delta_abs_mean": float(train_logp_delta_abs.mean().item()),
            "target_behavior_train_logp_delta_abs_p95": float(torch.quantile(train_logp_delta_abs, 0.95).item()),
            "target_behavior_train_logp_delta_abs_p99": float(torch.quantile(train_logp_delta_abs, 0.99).item()),
        }

    reward_metrics = rewards_for_metrics.detach()
    chosen_action_metrics = chosen_action_outcome_metrics(
        actions=actions,
        loss_mask=loss_mask,
        rewards=reward_metrics,
        advantages=advantages.detach(),
        action_catalog=action_catalog,
        pass_action_id=pass_action_id,
    )
    metrics = {
        "loss": float(total_loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_mean.detach()),
        "entropy_scope_family_active": float(entropy_scope == "family"),
        "policy_train_fraction": float(loss_mask.mean().detach()),
        "value_train_fraction": float(value_loss_mask.mean().detach()),
        "reward_mean": float(reward_metrics.mean().item()),
        "reward_std": float(reward_metrics.to(dtype=torch.float32).std(unbiased=False).item()),
        "reward_abs_mean": float(reward_metrics.abs().mean().item()),
        "reward_min": float(reward_metrics.min().item()),
        "reward_max": float(reward_metrics.max().item()),
        "reward_nonzero_fraction": float((reward_metrics != 0).float().mean().item()),
        "reward_positive_fraction": float((reward_metrics > 0).float().mean().item()),
        "reward_negative_fraction": float((reward_metrics < 0).float().mean().item()),
        "terminal_outcome_backfill_count": _float_or_zero(terminal_outcome_backfill_count),
        "terminal_outcome_backfill_total_micros": _float_or_zero(terminal_outcome_backfill_total_micros),
        "terminal_outcome_trace_backfill_count": _float_or_zero(terminal_outcome_trace_backfill_count),
        "terminal_outcome_trace_backfill_total_micros": _float_or_zero(terminal_outcome_trace_backfill_total_micros),
        "advantage_mean": float(advantages.detach().mean().item()),
        "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
        "target_mean": float(targets.detach().mean().item()),
        "target_abs_mean": float(targets.detach().abs().mean().item()),
        "vtrace_rho_mean": float(rho_metrics.mean().item()),
        "vtrace_rho_p50": float(torch.quantile(rho_metrics, 0.50).item()),
        "vtrace_rho_p90": float(torch.quantile(rho_metrics, 0.90).item()),
        "vtrace_rho_p95": float(torch.quantile(rho_metrics, 0.95).item()),
        "vtrace_rho_p99": float(torch.quantile(rho_metrics, 0.99).item()),
        "vtrace_train_rho_mean": float(train_rho_metrics.mean().item()),
        "vtrace_train_rho_p50": float(torch.quantile(train_rho_metrics, 0.50).item()),
        "vtrace_train_rho_p90": float(torch.quantile(train_rho_metrics, 0.90).item()),
        "vtrace_train_rho_p95": float(torch.quantile(train_rho_metrics, 0.95).item()),
        "vtrace_train_rho_p99": float(torch.quantile(train_rho_metrics, 0.99).item()),
        "vtrace_rho_clip_rate": float((rhos_for_metrics.detach() > rho_bar).float().mean().item()),
        "vtrace_c_clip_rate": float((rhos_for_metrics.detach() > c_bar).float().mean().item()),
        **logp_delta_metrics,
        **chosen_action_metrics,
        **trajectory_retention_metrics,
        **policy_anchor_metrics,
    }
    metrics.update(teacher_metrics)
    return metrics


__all__ = ["build_impala_loss_metrics", "chosen_action_outcome_metrics"]
