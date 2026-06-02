"""Prepared learner-batch inputs shared by runtime algorithms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.runtime.components.batching.bootstrap_values import (
    ImpalaBootstrapFields,
    RuntimeBootstrapFields,
    gae_advantages,
    impala_bootstrap_fields,
    reset_before_step,
    runtime_bootstrap_fields,
    runtime_discounts,
    runtime_done_flags,
)
from weiss_rl.runtime.components.batching.reward_backfill import TerminalBackfillMetrics, apply_runtime_reward_backfills
from weiss_rl.runtime.components.field_assembly import RuntimeBatchFields, concat_runtime_batch_fields

BootstrapFields = RuntimeBootstrapFields | ImpalaBootstrapFields


@dataclass(frozen=True, slots=True)
class SharedLearnerBatchInputs:
    fields: RuntimeBatchFields
    bootstrap: BootstrapFields
    done: np.ndarray
    discounts: np.ndarray
    reset_before_step: np.ndarray


@dataclass(frozen=True, slots=True)
class ImpalaLearnerBatchInputs:
    fields: RuntimeBatchFields
    bootstrap: ImpalaBootstrapFields
    rewards: np.ndarray
    discounts: np.ndarray
    reset_before_step: np.ndarray
    backfill_metrics: TerminalBackfillMetrics


@dataclass(frozen=True, slots=True)
class PpoLearnerBatchInputs:
    fields: RuntimeBatchFields
    bootstrap: RuntimeBootstrapFields
    discounts: np.ndarray
    reset_before_step: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


def prepare_shared_learner_batch_inputs(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    bootstrap: BootstrapFields,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> SharedLearnerBatchInputs:
    fields = concat_runtime_batch_fields(
        unrolls,
        action_dim=action_dim,
        record_batch_timer_ms=record_batch_timer_ms,
    )
    done = runtime_done_flags(terminated=fields.terminated, truncated=fields.truncated)
    discounts = runtime_discounts(
        done=done,
        to_play_seat=fields.to_play_seat,
        bootstrap_actor=bootstrap.actor,
        gamma=gamma,
    )
    return SharedLearnerBatchInputs(
        fields=fields,
        bootstrap=bootstrap,
        done=done,
        discounts=discounts,
        reset_before_step=reset_before_step(done),
    )


def prepare_impala_learner_batch_inputs(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    terminal_outcome_backfill_reward: float,
    terminal_outcome_trace_backfill_reward: float,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> ImpalaLearnerBatchInputs:
    bootstrap = impala_bootstrap_fields(unrolls)
    shared = prepare_shared_learner_batch_inputs(
        unrolls,
        action_dim=action_dim,
        record_batch_timer_ms=record_batch_timer_ms,
        bootstrap=bootstrap,
        gamma=gamma,
    )
    rewards, backfill_metrics = apply_runtime_reward_backfills(
        rewards=shared.fields.rewards,
        done=shared.done,
        policy_train_mask=shared.fields.policy_train_mask,
        terminal_outcome_backfill_reward=terminal_outcome_backfill_reward,
        terminal_outcome_trace_backfill_reward=terminal_outcome_trace_backfill_reward,
    )
    return ImpalaLearnerBatchInputs(
        fields=shared.fields,
        bootstrap=bootstrap,
        rewards=rewards,
        discounts=shared.discounts,
        reset_before_step=shared.reset_before_step,
        backfill_metrics=backfill_metrics,
    )


def prepare_ppo_learner_batch_inputs(
    unrolls: Sequence[Any],
    *,
    action_dim: int,
    gamma: float,
    gae_lambda: float,
    record_batch_timer_ms: Callable[[str, float], None] | None = None,
) -> PpoLearnerBatchInputs:
    bootstrap = runtime_bootstrap_fields(unrolls)
    shared = prepare_shared_learner_batch_inputs(
        unrolls,
        action_dim=action_dim,
        record_batch_timer_ms=record_batch_timer_ms,
        bootstrap=bootstrap,
        gamma=gamma,
    )
    advantages = gae_advantages(
        rewards=shared.fields.rewards,
        values=shared.fields.values,
        bootstrap_value=bootstrap.value,
        discounts=shared.discounts,
        gae_lambda=float(gae_lambda),
    )
    return PpoLearnerBatchInputs(
        fields=shared.fields,
        bootstrap=bootstrap,
        discounts=shared.discounts,
        reset_before_step=shared.reset_before_step,
        advantages=advantages,
        returns=advantages + shared.fields.values,
    )
