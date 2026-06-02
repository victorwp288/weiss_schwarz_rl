from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from weiss_rl.runtime.components.batching.learner_batch_inputs import (
    PpoLearnerBatchInputs,
    prepare_ppo_learner_batch_inputs,
)
from weiss_rl.runtime.components.field_assembly import base_runtime_learner_payload


def ppo_algorithm_payload(prepared: PpoLearnerBatchInputs) -> dict[str, Any]:
    return {
        "old_logp": prepared.fields.behavior_logp,
        "old_values": prepared.fields.values,
        "returns": prepared.returns,
        "advantages": prepared.advantages,
    }


def ppo_learner_payload(prepared: PpoLearnerBatchInputs) -> dict[str, Any]:
    return {
        **base_runtime_learner_payload(
            fields=prepared.fields,
            rewards=prepared.fields.rewards,
            discounts=prepared.discounts,
            reset_before_step=prepared.reset_before_step,
        ),
        **ppo_algorithm_payload(prepared),
    }


def build_ppo_learner_batch(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    gae_lambda: float,
    truncation_reward: float,
    truncation_bootstrap_value: bool,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    del truncation_reward, truncation_bootstrap_value
    prepared = prepare_ppo_learner_batch_inputs(
        unrolls,
        action_dim=action_dim,
        gamma=gamma,
        gae_lambda=gae_lambda,
        record_batch_timer_ms=record_batch_timer_ms,
    )
    return ppo_learner_payload(prepared)


__all__ = [
    "build_ppo_learner_batch",
    "ppo_algorithm_payload",
    "ppo_learner_payload",
]
