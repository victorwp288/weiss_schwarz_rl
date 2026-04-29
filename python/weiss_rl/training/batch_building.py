"""Learner-batch assembly helpers for the training entrypoint."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.masking import masked_logp_from_mask
from weiss_rl.model import PolicyValueModel

IMPALA_ALGORITHMS = frozenset({"impala_vtrace_gru", "impala_vtrace_ff", "structured_v2", "impala_vtrace_structured_v1"})
PPO_ALGORITHMS = frozenset({"ppo_lite_masked_v1"})


@dataclass(frozen=True, slots=True)
class MinimalRollout:
    obs: np.ndarray
    legal_mask: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    logits: np.ndarray
    values: np.ndarray
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray


def bootstrap_values(
    model: PolicyValueModel,
    rollout: MinimalRollout,
    final_seat_hidden: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    bootstrap_value = np.zeros((rollout.bootstrap_obs.shape[0],), dtype=np.float32)
    valid_rows = (rollout.bootstrap_actor == 0) | (rollout.bootstrap_actor == 1)
    if not np.any(valid_rows):
        return bootstrap_value

    with torch.inference_mode():
        _, value_tensor, _ = model.forward_seat_aware(
            torch.as_tensor(rollout.bootstrap_obs[valid_rows], device=device),
            torch.as_tensor(rollout.bootstrap_actor[valid_rows], device=device, dtype=torch.long),
            final_seat_hidden[valid_rows],
        )
    bootstrap_value[valid_rows] = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return bootstrap_value


def build_learner_batch(
    stack: StackConfig,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    *,
    action_dim: int,
    initial_hidden_state: torch.Tensor,
    pass_action_id: int,
) -> dict[str, Any]:
    training_config = stack.config.training
    rewards_config = stack.config.rewards
    if training_config is None or rewards_config is None:
        raise RuntimeError("The canonical single-node path requires training and rewards config blocks")

    target_logp = masked_logp_from_mask(
        rollout.logits.reshape(-1, action_dim),
        rollout.legal_mask.reshape(-1, action_dim),
        rollout.actions.reshape(-1),
        pass_action_id=pass_action_id,
    ).reshape(rollout.actions.shape)

    rewards = np.asarray(rollout.rewards, dtype=np.float32)

    discounts = np.logical_not(rollout.terminated).astype(np.float32) * float(rewards_config.gamma)
    if not bool(rewards_config.truncation.bootstrap_value):
        discounts *= np.logical_not(rollout.truncated).astype(np.float32)

    values = np.concatenate([rollout.values, bootstrap_value[np.newaxis, :]], axis=0)
    vtrace_result: VTraceTargets = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        rollout.behavior_logp,
        target_logp,
        rho_bar=training_config.vtrace_rho_bar,
        c_bar=training_config.vtrace_c_bar,
    )

    return {
        "obs": rollout.obs,
        "actions": rollout.actions,
        "legal_mask": rollout.legal_mask,
        "to_play_seat": rollout.to_play_seat,
        "actor": rollout.to_play_seat,
        "initial_hidden_state": initial_hidden_state.detach().cpu().numpy(),
        "rewards": rewards,
        "discounts": discounts,
        "behavior_logp": rollout.behavior_logp,
        "behavior_logits": rollout.logits,
        "logits": rollout.logits,
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": float(training_config.vtrace_rho_bar),
        "vtrace_c_bar": float(training_config.vtrace_c_bar),
    }


def pass_with_nonpass_penalty_from_rewards_config(rewards_config: Any) -> float:
    pass_with_nonpass_penalty = float(
        getattr(getattr(rewards_config, "shaping", None), "pass_with_nonpass_penalty", 0.0)
    )
    if str(getattr(rewards_config, "objective", "")).strip().lower() == "terminal_only_pm1":
        return 0.0
    return pass_with_nonpass_penalty


def collect_training_batch(
    *,
    runtime: Any,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
) -> Any:
    pass_with_nonpass_penalty = pass_with_nonpass_penalty_from_rewards_config(rewards_config)
    if algorithm in IMPALA_ALGORITHMS:
        return runtime.collect_update_batch(
            gamma=float(rewards_config.gamma),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            pass_with_nonpass_penalty=pass_with_nonpass_penalty,
            vtrace_rho_bar=float(training_config.vtrace_rho_bar),
            vtrace_c_bar=float(training_config.vtrace_c_bar),
        )
    if algorithm in PPO_ALGORITHMS:
        return runtime.collect_policy_batch(
            gamma=float(rewards_config.gamma),
            gae_lambda=float(training_config.ppo_gae_lambda),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            pass_with_nonpass_penalty=pass_with_nonpass_penalty,
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


@contextmanager
def torch_num_threads_scope(num_threads: int | None) -> Iterator[None]:
    if num_threads is None:
        yield
        return
    target = int(num_threads)
    if target < 1:
        raise ValueError("num_threads must be >= 1")
    previous = int(torch.get_num_threads())
    if previous == target:
        yield
        return
    torch.set_num_threads(target)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def collect_training_batch_prefetch(
    *,
    runtime: Any,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    actor_torch_threads: int | None,
) -> Any:
    with torch_num_threads_scope(actor_torch_threads):
        return collect_training_batch(
            runtime=runtime,
            algorithm=algorithm,
            training_config=training_config,
            rewards_config=rewards_config,
        )
