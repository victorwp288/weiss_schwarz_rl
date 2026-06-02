from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weiss_rl.core.masking import masked_logp_from_mask
from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_targets
from weiss_rl.runtime.components.batching import actor_perspective_discounts
from weiss_rl.training.algorithm_families import (
    IMPALA_ALGORITHMS as IMPALA_ALGORITHMS,
)
from weiss_rl.training.algorithm_families import (
    PPO_ALGORITHMS as PPO_ALGORITHMS,
)
from weiss_rl.training.algorithm_families import (
    training_algorithm_family,
)


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


@dataclass(frozen=True, slots=True)
class _ImpalaBatchInputs:
    rewards: np.ndarray
    discounts: np.ndarray
    reset_before_step: np.ndarray
    vtrace_result: VTraceTargets


def bootstrap_values(
    model: Any,
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


def collect_training_batch(
    *,
    runtime: Any,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
) -> Any:
    """Collect one algorithm-specific training batch from the queue runtime."""

    family = training_algorithm_family(algorithm)
    if family == "impala":
        return runtime.collect_update_batch(
            gamma=float(rewards_config.gamma),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
            vtrace_rho_bar=float(training_config.vtrace_rho_bar),
            vtrace_c_bar=float(training_config.vtrace_c_bar),
        )
    if family == "ppo":
        return runtime.collect_policy_batch(
            gamma=float(rewards_config.gamma),
            gae_lambda=float(training_config.ppo_gae_lambda),
            truncation_reward=float(rewards_config.truncation.reward),
            truncation_bootstrap_value=bool(rewards_config.truncation.bootstrap_value),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _training_and_rewards_config(stack: Any) -> tuple[Any, Any]:
    training_config = stack.config.training
    rewards_config = stack.config.rewards
    if training_config is None or rewards_config is None:
        raise RuntimeError("The canonical single-node path requires training and rewards config blocks")
    return training_config, rewards_config


def _rollout_target_logp(
    rollout: MinimalRollout,
    *,
    action_dim: int,
    pass_action_id: int,
) -> np.ndarray:
    return masked_logp_from_mask(
        rollout.logits.reshape(-1, action_dim),
        rollout.legal_mask.reshape(-1, action_dim),
        rollout.actions.reshape(-1),
        pass_action_id=pass_action_id,
    ).reshape(rollout.actions.shape)


def _rollout_done_flags(rollout: MinimalRollout) -> np.ndarray:
    return np.logical_or(rollout.terminated, rollout.truncated)


def _reset_before_step(done: np.ndarray) -> np.ndarray:
    reset_before_step = np.zeros_like(done, dtype=np.bool_)
    reset_before_step[1:] = done[:-1]
    return reset_before_step


def _values_with_bootstrap(rollout: MinimalRollout, bootstrap_value: np.ndarray) -> np.ndarray:
    return np.concatenate([rollout.values, bootstrap_value[np.newaxis, :]], axis=0)


def _impala_batch_inputs(
    *,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    action_dim: int,
    pass_action_id: int,
    gamma: float,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
) -> _ImpalaBatchInputs:
    target_logp = _rollout_target_logp(
        rollout,
        action_dim=action_dim,
        pass_action_id=pass_action_id,
    )
    rewards = np.asarray(rollout.rewards, dtype=np.float32)
    done = _rollout_done_flags(rollout)
    discounts = actor_perspective_discounts(
        done=done,
        to_play_seat=rollout.to_play_seat,
        bootstrap_actor=rollout.bootstrap_actor,
        gamma=gamma,
    )
    values = _values_with_bootstrap(rollout, bootstrap_value)
    vtrace_result = compute_vtrace_targets(
        rewards,
        values,
        discounts,
        rollout.behavior_logp,
        target_logp,
        rho_bar=vtrace_rho_bar,
        c_bar=vtrace_c_bar,
    )
    return _ImpalaBatchInputs(
        rewards=rewards,
        discounts=discounts,
        reset_before_step=_reset_before_step(done),
        vtrace_result=vtrace_result,
    )


def _impala_learner_payload(
    *,
    rollout: MinimalRollout,
    inputs: _ImpalaBatchInputs,
    initial_hidden_state: torch.Tensor,
    vtrace_rho_bar: float,
    vtrace_c_bar: float,
) -> dict[str, Any]:
    return {
        "obs": rollout.obs,
        "actions": rollout.actions,
        "legal_mask": rollout.legal_mask,
        "to_play_seat": rollout.to_play_seat,
        "actor": rollout.to_play_seat,
        "initial_hidden_state": initial_hidden_state.detach().cpu().numpy(),
        "rewards": inputs.rewards,
        "discounts": inputs.discounts,
        "reset_before_step": inputs.reset_before_step,
        "behavior_logp": rollout.behavior_logp,
        "behavior_logits": rollout.logits,
        "logits": rollout.logits,
        "vtrace_result": inputs.vtrace_result,
        "vtrace_rho_bar": float(vtrace_rho_bar),
        "vtrace_c_bar": float(vtrace_c_bar),
    }


def build_learner_batch(
    stack: Any,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    *,
    action_dim: int,
    initial_hidden_state: torch.Tensor,
    pass_action_id: int,
) -> dict[str, Any]:
    training_config, rewards_config = _training_and_rewards_config(stack)
    inputs = _impala_batch_inputs(
        rollout=rollout,
        bootstrap_value=bootstrap_value,
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        gamma=float(rewards_config.gamma),
        vtrace_rho_bar=float(training_config.vtrace_rho_bar),
        vtrace_c_bar=float(training_config.vtrace_c_bar),
    )
    return _impala_learner_payload(
        rollout=rollout,
        inputs=inputs,
        initial_hidden_state=initial_hidden_state,
        vtrace_rho_bar=float(training_config.vtrace_rho_bar),
        vtrace_c_bar=float(training_config.vtrace_c_bar),
    )
